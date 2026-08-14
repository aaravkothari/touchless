"""Main loop wiring: tracking -> calibration map -> smoothing -> cursor.

Three modes:
  preview   - visualize tracking, no cursor control (safe playground)
  calibrate - 3-posture dot sequence + validation, writes calibration.json
  run       - actually drive the cursor (requires calibration first)
"""

import csv
import os
import time
from collections import deque

import cv2
import numpy as np
import pyautogui

from .calibration import run_calibration
from .camera import Camera
from .clicker import BlinkClicker, DwellClicker, PinchHold
from .config import Config
from .hands import HandTracker
from .model import GazeModel
from .mouse import Cursor
from .pursuit import PursuitData
from .smoothing import OneEuro2D
from .tracking import FaceSample, FaceTracker

_HUD = cv2.FONT_HERSHEY_SIMPLEX
# Eye corners + irises, drawn in the preview overlay.
_HUD_LANDMARKS = (33, 133, 159, 145, 468, 362, 263, 386, 374, 473)
# Depth bar range (cm) — sitting distances land inside this comfortably.
_DEPTH_LO, _DEPTH_HI = 20.0, 100.0


def _telemetry(s, cfg, fps, status):
    """The 'everything we track' readout: gaze, head angles, position, depth."""
    lines = [f"{status}   {fps:4.1f} fps"]
    if not s.ok:
        lines.append("NO FACE")
        return lines
    blink_state = "CLOSED" if s.blink > cfg.blink_gate else "open"
    lines += [
        f"gaze  x {s.gaze_x:+.3f}  y {s.gaze_y:+.3f}   blink {s.blink:.2f} {blink_state}",
        f"head  yaw {s.yaw:+5.1f}  pitch {s.pitch:+5.1f}  roll {s.roll:+5.1f}",
        f"pos   x {s.tx:+5.1f}cm  y {s.ty:+5.1f}cm  depth {s.depth:4.1f}cm",
    ]
    return lines


def _draw_hud(frame, lines, pred=None, scale=1.0, points=(), depth=None):
    """Mirror + optionally downscale the frame, overlay tracking info.

    points: landmark pixel coords (in frame space) to dot in green.
    pred: normalized (x, y) prediction, drawn in a mini screen-rect (top
    right) so you can see where the cursor would go.
    depth: if given, a lean-in/lean-out bar is drawn under the text.
    """
    view = cv2.flip(frame, 1)
    if scale != 1.0:
        view = cv2.resize(view, (int(view.shape[1] * scale), int(view.shape[0] * scale)))
    h, w = view.shape[:2]
    for x, y in points:
        cv2.circle(view, (w - 1 - int(x * scale), int(y * scale)), 2, (0, 220, 0), -1)
    for n, line in enumerate(lines):
        cv2.putText(view, line, (10, 24 + n * 24), _HUD, 0.55, (0, 220, 220), 1)
    if depth is not None:
        y0 = 24 + len(lines) * 24
        frac = float(np.clip((depth - _DEPTH_LO) / (_DEPTH_HI - _DEPTH_LO), 0, 1))
        cv2.rectangle(view, (10, y0), (10 + 150, y0 + 10), (80, 80, 80), 1)
        cv2.rectangle(view, (10, y0), (10 + int(150 * frac), y0 + 10), (0, 220, 220), -1)
        cv2.putText(view, "depth", (168, y0 + 10), _HUD, 0.45, (150, 150, 150), 1)
    if pred is not None:
        rw, rh = w // 4, h // 4
        x0 = w - rw - 10
        cv2.rectangle(view, (x0, 10), (x0 + rw, 10 + rh), (80, 80, 80), 1)
        px = x0 + int(np.clip(pred[0], 0, 1) * rw)
        py = 10 + int(np.clip(pred[1], 0, 1) * rh)
        cv2.drawMarker(view, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
    return view


def _face_hud_extras(s):
    """(points, depth) for _draw_hud from a FaceSample."""
    if not s.ok or s.landmarks is None:
        return (), None
    return s.landmarks[list(_HUD_LANDMARKS)], s.depth


def _hand_telemetry(s, face, fps, status, held=None, wrist=False):
    lines = [f"{status}   {fps:4.1f} fps"]
    if s.pointer_ok:
        if wrist:
            lines.append(f"RIGHT tip-wrist ({s.pointer_rel[0]:+.2f}, "
                         f"{s.pointer_rel[1]:+.2f})  <- moves (wrist-relative)")
        else:
            lines.append(f"RIGHT pointer ({s.pointer[0]:.2f}, {s.pointer[1]:.2f})  <- moves")
    else:
        lines.append("RIGHT hand: NOT FOUND (point with your right index finger)")
    if s.left_ok:
        lines.append(f"LEFT  pinch index {s.pinch_index:.2f}  middle {s.pinch_middle:.2f}"
                     + (f"   [{held.upper()}-CLICK HELD]" if held else ""))
    else:
        lines.append("LEFT  hand: not in view (it's the click hand)")
    if face is not None and face.ok:
        lines.append(f"FACE  jaw {face.jaw:.2f}  tongue {face.tongue:.2f}"
                     + ("  <- RECENTER!" if face.tongue > 0.45 else ""))
    else:
        lines.append("FACE  not in view (tongue = recenter)")
    return lines


def _hand_points(s, show_ref=False):
    pts = [tuple(p) for _, lm in s.hands for p in lm]
    if show_ref and s.ref_px is not None:
        pts.append(tuple(s.ref_px))  # virtual forearm reference dot
    return pts


def _load_model(cfg: Config) -> GazeModel | None:
    if not os.path.exists(cfg.model_file):
        return None
    try:
        return GazeModel.load(cfg.model_file)
    except ValueError as e:
        print(e)
        return None


class _Fps:
    def __init__(self):
        self.t = deque(maxlen=30)

    def tick(self) -> float:
        self.t.append(time.monotonic())
        if len(self.t) < 2:
            return 0.0
        return (len(self.t) - 1) / (self.t[-1] - self.t[0])


def preview(cfg: Config, input_mode: str = "gaze"):
    """Show tracking output. Use this to verify lighting/camera before calibrating."""
    if input_mode in ("hand", "hand-wrist"):
        _preview_hand(cfg, wrist=input_mode == "hand-wrist")
        return
    model = _load_model(cfg)  # optional: shows live prediction if calibrated
    tracker = FaceTracker(cfg)
    fps = _Fps()
    try:
        while True:
            frame, s = tracker.read()
            if frame is None:
                continue
            lines = _telemetry(s, cfg, fps.tick(), "preview - q to quit")
            pred = model.predict(s.features) if (model is not None and s.ok) else None
            if pred is not None:
                lines.append(f"pred  ({pred[0]:+.2f}, {pred[1]:+.2f})")
            points, depth = _face_hud_extras(s)
            cv2.imshow("touchless preview",
                       _draw_hud(frame, lines, pred, points=points, depth=depth))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


class _HandStack:
    """Shared camera + hand landmarker every frame + face landmarker every
    Nth frame (tongue detection doesn't need full rate)."""

    def __init__(self, cfg: Config):
        from dataclasses import replace
        cam_cfg = replace(cfg, frame_width=cfg.hand_frame_width,
                          frame_height=cfg.hand_frame_height)
        self.camera = Camera(cam_cfg)
        self.hand = HandTracker(cfg, self.camera)
        self.face = FaceTracker(cfg, self.camera)
        self.every_n = max(cfg.face_every_n, 1)
        self._i = 0
        self._last_face = FaceSample()

    def read(self):
        frame = self.camera.read()
        if frame is None:
            return None, None, self._last_face
        hs = self.hand.process(frame)
        if self._i % self.every_n == 0:
            self._last_face = self.face.process(frame)
        self._i += 1
        return frame, hs, self._last_face

    def close(self):
        self.hand.close()
        self.face.close()
        self.camera.close()


def _preview_hand(cfg: Config, wrist: bool = False):
    stack = _HandStack(cfg)
    fps = _Fps()
    label = "hand-wrist preview" if wrist else "hand preview"
    try:
        while True:
            frame, s, face = stack.read()
            if frame is None:
                continue
            lines = _hand_telemetry(s, face, fps.tick(), f"{label} - q to quit",
                                    wrist=wrist)
            cv2.imshow("touchless preview",
                       _draw_hud(frame, lines, points=_hand_points(s, show_ref=wrist)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stack.close()
        cv2.destroyAllWindows()


def calibrate(cfg: Config):
    screen_w, screen_h = pyautogui.size()
    model = run_calibration(cfg, screen_w, screen_h)
    if model is None:
        print("Calibration aborted, nothing saved.")
        return
    model.save(cfg.model_file)
    print(f"Model ({model.name}) saved to {cfg.model_file}")


def retrain(cfg: Config):
    """Refit the model from the recorded pursuit session - no camera needed."""
    if not os.path.exists(cfg.data_file):
        print(f"No {cfg.data_file} found - run a calibration first to record one.")
        return
    data = PursuitData.load(cfg.data_file)
    screen_w, screen_h = pyautogui.size()
    model = GazeModel.fit(data, cfg, screen_w, screen_h)
    model.save(cfg.model_file)
    print(f"Model ({model.name}) saved to {cfg.model_file}")


def run(cfg: Config, click_mode: str, log_path: str | None = None,
        input_mode: str = "gaze"):
    if input_mode in ("hand", "hand-wrist"):
        if click_mode != "off":
            print("note: hand mode has a fixed control scheme (left-hand "
                  "pinches click, tongue recenters) - ignoring --click")
        _run_hand(cfg, wrist=input_mode == "hand-wrist")
        return
    if click_mode == "pinch":
        print("--click pinch is part of hand mode; gaze mode supports "
              "--click off | dwell | blink")
        return
    model = _load_model(cfg)
    if model is None:
        print(f"No usable {cfg.model_file} - run calibration first:")
        print("  python -m touchless calibrate")
        return
    print(f"Running, click={click_mode}.")
    print("Controls: q quits, space pauses (preview window must be focused).")
    print("Emergency stop: slam the mouse into the top-left corner.")

    tracker = FaceTracker(cfg)
    cursor = Cursor(cfg.screen_inset_px)
    smoother = OneEuro2D(cfg.smooth_min_cutoff, cfg.smooth_beta)
    dwell = DwellClicker(cfg) if click_mode == "dwell" else None
    blink = BlinkClicker(cfg) if click_mode == "blink" else None
    fps = _Fps()
    raw_hist: deque[tuple[float, float]] = deque(maxlen=3)  # spike pre-filter
    paused = False

    log_fh = log_writer = None
    if log_path:
        log_fh = open(log_path, "w", newline="")
        log_writer = csv.writer(log_fh)
        log_writer.writerow(["t", "gaze_x", "gaze_y", "yaw", "pitch", "roll",
                             "tx", "ty", "tz", "blink",
                             "raw_x", "raw_y", "smooth_x", "smooth_y"])

    try:
        while True:
            frame, s = tracker.read()
            if frame is None:
                continue

            status = "PAUSED (space to resume)" if paused else "LIVE - q quit, space pause"
            pred = None
            blinking = s.ok and s.blink > cfg.blink_gate
            if s.ok and not paused and not blinking:
                # Blink gating: while eyes are closed the gaze blendshapes go
                # wild, so the cursor holds position instead of jumping.
                nx, ny = model.predict(s.features)
                # Clamp before smoothing so an off-screen fling can't wind up
                # the filter, then median-of-3 to kill single-frame spikes.
                nx = float(np.clip(nx, -cfg.pred_clamp, 1 + cfg.pred_clamp))
                ny = float(np.clip(ny, -cfg.pred_clamp, 1 + cfg.pred_clamp))
                raw_hist.append((nx, ny))
                mx = float(np.median([p[0] for p in raw_hist]))
                my = float(np.median([p[1] for p in raw_hist]))
                sx, sy = smoother.apply(mx, my, time.monotonic())
                cursor.move_norm(sx, sy)
                pred = (sx, sy)
                if log_writer is not None:
                    log_writer.writerow(
                        [f"{time.monotonic():.3f}",
                         *(f"{v:.4f}" for v in s.features), f"{s.blink:.3f}",
                         f"{nx:.4f}", f"{ny:.4f}", f"{sx:.4f}", f"{sy:.4f}"])
                if dwell is not None:
                    px, py = cursor.position()
                    if dwell.update(px, py):
                        cursor.click()
            if s.ok and not paused and blink is not None and blink.update(s.blink):
                cursor.click()

            lines = _telemetry(s, cfg, fps.tick(), status)
            if s.ok and dwell is not None and dwell.progress() > 0:
                lines.append("dwell " + "#" * int(dwell.progress() * 10))
            points, depth = _face_hud_extras(s)
            cv2.imshow("touchless", _draw_hud(frame, lines, pred, scale=0.5,
                                              points=points, depth=depth))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                smoother.reset()
                raw_hist.clear()
    except pyautogui.FailSafeException:
        print("Fail-safe triggered (mouse in top-left corner). Stopped.")
    finally:
        if log_fh is not None:
            log_fh.close()
            print(f"Session log written to {log_path}")
        tracker.close()
        cv2.destroyAllWindows()


def _run_hand(cfg: Config, wrist: bool = False):
    """Hand mode: cursor = center + gain * (right index tip - anchor).

    x is flipped so moving your hand right moves the cursor right on an
    unmirrored frame. Controls: LEFT thumb+index pinch = left button,
    LEFT thumb+middle pinch = right button (held pinch = held button),
    tongue out = recenter + re-anchor.

    wrist=True (hand-wrist mode): identical scheme, but the pointer is the
    index tip RELATIVE to the wrist (hand-size units), so translating the
    whole hand holds position and only finger articulation moves the cursor.
    """
    if wrist:
        print("Hand-wrist mode. RIGHT index finger RELATIVE TO WRIST moves "
              "the cursor - moving the whole hand does nothing.")
    else:
        print("Hand mode. RIGHT index finger moves the cursor.")
    print("LEFT thumb+index pinch = left click (hold to drag).")
    print("LEFT thumb+middle pinch = right click.")
    print("Stick your TONGUE out to recenter the cursor.")
    print("Controls: q quits, space pauses (preview window must be focused).")
    print("Emergency stop: slam the mouse into the top-left corner.")

    stack = _HandStack(cfg)
    cursor = Cursor(cfg.screen_inset_px)
    gain_x = cfg.hand_wrist_gain_x if wrist else cfg.hand_gain_x
    gain_y = cfg.hand_wrist_gain_y if wrist else cfg.hand_gain_y
    if wrist:
        smoother = OneEuro2D(cfg.hand_wrist_smooth_min_cutoff,
                             cfg.hand_wrist_smooth_beta)
    else:
        smoother = OneEuro2D(cfg.hand_smooth_min_cutoff, cfg.hand_smooth_beta)
    screen_w, screen_h = pyautogui.size()
    dead_x = cfg.hand_deadzone_px / screen_w
    dead_y = cfg.hand_deadzone_px / screen_h
    sent: tuple[float, float] | None = None  # last position actually forwarded
    left_pinch = PinchHold(cfg)
    right_pinch = PinchHold(cfg)
    fps = _Fps()
    raw_hist: deque[tuple[float, float]] = deque(maxlen=3)  # spike pre-filter
    anchor: np.ndarray | None = None
    tongue_since: float | None = None
    last_recenter = 0.0
    held: str | None = None   # which mouse button is currently down
    paused = False

    def press(button: str):
        nonlocal held
        cursor.down(button)
        held = button

    def release():
        nonlocal held
        if held is not None:
            cursor.up(held)
            held = None

    try:
        while True:
            frame, s, face = stack.read()
            if frame is None:
                continue
            now = time.monotonic()

            status = "PAUSED (space to resume)" if paused else "LIVE - q quit, space pause"
            pred = None
            if not paused and s.pointer_ok:
                pointer = s.pointer_rel if wrist else s.pointer
                if anchor is None:
                    anchor = pointer.copy()  # first sighting = neutral

                # Tongue out (held briefly) -> recenter and re-anchor.
                if face.ok and face.tongue > cfg.tongue_threshold:
                    if tongue_since is None:
                        tongue_since = now
                    if (now - tongue_since >= cfg.tongue_hold_s
                            and now - last_recenter >= cfg.tongue_cooldown_s):
                        anchor = pointer.copy()
                        smoother.reset()
                        raw_hist.clear()
                        cursor.move_norm(0.5, 0.5)
                        sent = (0.5, 0.5)
                        last_recenter = now
                else:
                    tongue_since = None

                d = pointer - anchor
                nx = 0.5 - gain_x * float(d[0])  # x flipped: mirror-natural
                ny = 0.5 + gain_y * float(d[1])
                nx = float(np.clip(nx, -cfg.pred_clamp, 1 + cfg.pred_clamp))
                ny = float(np.clip(ny, -cfg.pred_clamp, 1 + cfg.pred_clamp))
                raw_hist.append((nx, ny))
                mx = float(np.median([p[0] for p in raw_hist]))
                my = float(np.median([p[1] for p in raw_hist]))
                sx, sy = smoother.apply(mx, my, now)
                # Deadzone with hysteresis: sub-pixel-scale wobble around the
                # last sent position is swallowed; a slow drift accumulates
                # until it clears the threshold and then goes through.
                if (sent is None or abs(sx - sent[0]) > dead_x
                        or abs(sy - sent[1]) > dead_y):
                    cursor.move_norm(sx, sy)
                    sent = (sx, sy)
                pred = (sx, sy)
            else:
                tongue_since = None  # pointer lost: hold position, reset gesture

            # Left-hand pinches -> mouse buttons. Lost hand = huge pinch
            # value, so a held button always releases. While one button is
            # held the other pinch is ignored (a full-hand grab can't fire
            # both).
            if not paused:
                pi = s.pinch_index if s.left_ok else 9.9
                pm = s.pinch_middle if s.left_ok else 9.9
                ev_l = left_pinch.update(pi if held in (None, "left") else 9.9)
                ev_r = right_pinch.update(pm if held in (None, "right") else 9.9)
                if ev_l == "down" and held is None:
                    press("left")
                elif ev_l == "up" and held == "left":
                    release()
                if ev_r == "down" and held is None:
                    press("right")
                elif ev_r == "up" and held == "right":
                    release()

            lines = _hand_telemetry(s, face, fps.tick(), status, held, wrist)
            cv2.imshow("touchless", _draw_hud(frame, lines, pred, scale=0.5,
                                              points=_hand_points(s, show_ref=wrist)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                release()
                smoother.reset()
                raw_hist.clear()
                sent = None
    except pyautogui.FailSafeException:
        print("Fail-safe triggered (mouse in top-left corner). Stopped.")
    finally:
        try:
            release()  # never leave a mouse button stuck down
        except Exception:
            pass
        stack.close()
        cv2.destroyAllWindows()
