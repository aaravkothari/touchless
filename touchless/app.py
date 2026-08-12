"""Main loop wiring: tracking -> calibration map -> smoothing -> cursor.

Three modes:
  preview   - visualize tracking, no cursor control (safe playground)
  calibrate - fullscreen dot sequence + validation, writes calibration.json
  run       - actually drive the cursor (requires calibration first)
"""

import csv
import os
import time
from collections import deque

import cv2
import numpy as np
import pyautogui

from .calibration import CalibrationMap, run_calibration
from .clicker import BlinkClicker, DwellClicker
from .config import Config
from .mouse import Cursor
from .smoothing import OneEuro2D
from .tracking import FaceTracker

_HUD = cv2.FONT_HERSHEY_SIMPLEX
# Eye corners + irises, drawn in the preview overlay.
_HUD_LANDMARKS = (33, 133, 159, 145, 468, 362, 263, 386, 374, 473)


def _draw_hud(frame, sample, lines, pred=None):
    """Mirror the frame for display, overlay tracking info.

    pred: optional normalized (x, y) prediction, drawn in a mini screen-rect
    in the top-right corner so you can see where the cursor would go.
    """
    view = cv2.flip(frame, 1)
    h, w = view.shape[:2]
    if sample.ok and sample.landmarks is not None:
        for i in _HUD_LANDMARKS:
            x, y = sample.landmarks[i]
            cv2.circle(view, (w - 1 - int(x), int(y)), 2, (0, 220, 0), -1)
    for n, line in enumerate(lines):
        cv2.putText(view, line, (10, 24 + n * 24), _HUD, 0.6, (0, 220, 220), 1)
    if pred is not None:
        rw, rh = w // 4, h // 4
        x0, y0 = w - rw - 10, 10
        cv2.rectangle(view, (x0, y0), (x0 + rw, y0 + rh), (80, 80, 80), 1)
        px = x0 + int(np.clip(pred[0], 0, 1) * rw)
        py = y0 + int(np.clip(pred[1], 0, 1) * rh)
        cv2.drawMarker(view, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
    return view


def _load_map(cfg: Config) -> CalibrationMap | None:
    if not os.path.exists(cfg.calib_file):
        return None
    try:
        return CalibrationMap.load(cfg.calib_file)
    except ValueError as e:
        print(e)
        return None


def preview(cfg: Config):
    """Show tracking output. Use this to verify lighting/camera before calibrating."""
    cmap = _load_map(cfg)  # optional: shows live prediction if calibrated
    tracker = FaceTracker(cfg)
    fps_t, fps = deque(maxlen=30), 0.0
    try:
        while True:
            frame, s = tracker.read()
            if frame is None:
                continue
            fps_t.append(time.monotonic())
            if len(fps_t) > 1:
                fps = (len(fps_t) - 1) / (fps_t[-1] - fps_t[0])
            lines = [f"preview - q to quit   {fps:4.1f} fps"]
            pred = None
            if s.ok:
                lines += [
                    f"gaze  x {s.gaze_x:+.3f}  y {s.gaze_y:+.3f}",
                    f"head  yaw {s.yaw:+5.1f}  pitch {s.pitch:+5.1f}",
                    f"blink {s.blink:.2f}  ({'CLOSED' if s.blink > cfg.blink_closed_threshold else 'open'})",
                ]
                if cmap is not None:
                    pred = cmap.predict(s.features)
                    lines.append(f"pred  ({pred[0]:+.2f}, {pred[1]:+.2f})")
            else:
                lines.append("NO FACE")
            cv2.imshow("touchless preview", _draw_hud(frame, s, lines, pred))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


def calibrate(cfg: Config):
    screen_w, screen_h = pyautogui.size()
    cmap = run_calibration(cfg, screen_w, screen_h)
    if cmap is None:
        print("Calibration aborted, nothing saved.")
        return
    cmap.save(cfg.calib_file)
    print(f"Calibration saved to {cfg.calib_file}")


def run(cfg: Config, click_mode: str, log_path: str | None = None):
    cmap = _load_map(cfg)
    if cmap is None:
        print(f"No usable {cfg.calib_file} - run calibration first:")
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
    raw_hist: deque[tuple[float, float]] = deque(maxlen=3)  # spike pre-filter
    paused = False

    log_fh = log_writer = None
    if log_path:
        log_fh = open(log_path, "w", newline="")
        log_writer = csv.writer(log_fh)
        log_writer.writerow(["t", "gaze_x", "gaze_y", "yaw", "pitch", "tx", "ty",
                             "raw_x", "raw_y", "smooth_x", "smooth_y"])

    try:
        while True:
            frame, s = tracker.read()
            if frame is None:
                continue

            status = "PAUSED (space to resume)" if paused else "LIVE"
            pred = None
            if s.ok and not paused:
                nx, ny = cmap.predict(s.features)
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
                        [f"{time.monotonic():.3f}", *(f"{v:.4f}" for v in s.features),
                         f"{nx:.4f}", f"{ny:.4f}", f"{sx:.4f}", f"{sy:.4f}"])
                if dwell is not None:
                    px, py = cursor.position()
                    if dwell.update(px, py):
                        cursor.click()
                if blink is not None and blink.update(s.blink):
                    cursor.click()

            lines = [f"{status} - q quit, space pause"]
            if not s.ok:
                lines.append("NO FACE")
            elif dwell is not None and dwell.progress() > 0:
                lines.append("dwell " + "#" * int(dwell.progress() * 10))
            view = _draw_hud(frame, s, lines, pred)
            cv2.imshow("touchless", cv2.resize(view, (view.shape[1] // 2, view.shape[0] // 2)))

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
