"""Main loop wiring: tracking -> calibration map -> smoothing -> cursor.

Three modes:
  preview   - visualize tracking, no cursor control (safe playground)
  calibrate - fullscreen dot sequence, writes calibration.json
  run       - actually drive the cursor (requires calibration first)
"""

import os
import time

import cv2
import numpy as np
import pyautogui

from .calibration import CalibrationMap, features, run_calibration
from .clicker import BlinkClicker, DwellClicker
from .config import Config
from .mouse import Cursor
from .smoothing import OneEuro2D
from .tracking import FaceTracker

_HUD = cv2.FONT_HERSHEY_SIMPLEX


def _draw_hud(frame, sample, lines):
    """Mirror the frame for display and overlay tracking info."""
    view = cv2.flip(frame, 1)
    h, w = view.shape[:2]
    if sample.ok and sample.landmarks is not None:
        for i in (33, 133, 159, 145, 468, 362, 263, 386, 374, 473):
            x, y = sample.landmarks[i]
            cv2.circle(view, (w - 1 - int(x), int(y)), 2, (0, 220, 0), -1)
    for n, line in enumerate(lines):
        cv2.putText(view, line, (10, 24 + n * 24), _HUD, 0.6, (0, 220, 220), 1)
    return view


def preview(cfg: Config):
    """Show tracking output. Use this to verify lighting/camera before calibrating."""
    tracker = FaceTracker(cfg)
    try:
        while True:
            frame, s = tracker.read()
            if frame is None:
                continue
            lines = ["preview - q to quit"]
            if s.ok:
                lines += [
                    f"gaze  x {s.gaze_x:+.3f}  y {s.gaze_y:+.3f}",
                    f"head  yaw {s.yaw:+5.1f}  pitch {s.pitch:+5.1f}",
                    f"ear   {s.ear:.3f}  ({'CLOSED' if s.ear < cfg.ear_closed_threshold else 'open'})",
                ]
            else:
                lines.append("NO FACE")
            cv2.imshow("touchless preview", _draw_hud(frame, s, lines))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


def calibrate(cfg: Config, mode: str):
    screen_w, screen_h = pyautogui.size()
    cmap = run_calibration(cfg, mode, screen_w, screen_h)
    if cmap is None:
        print("Calibration aborted.")
        return
    cmap.save(cfg.calib_file)
    print(f"Calibration ({mode} mode) saved to {cfg.calib_file}")


def run(cfg: Config, click_mode: str):
    if not os.path.exists(cfg.calib_file):
        print(f"No {cfg.calib_file} found - run calibration first:")
        print("  python -m touchless calibrate")
        return
    cmap = CalibrationMap.load(cfg.calib_file)
    print(f"Running in {cmap.mode} mode, click={click_mode}.")
    print("Controls: q quits, space pauses (preview window must be focused).")
    print("Emergency stop: slam the mouse into the top-left corner.")

    tracker = FaceTracker(cfg)
    cursor = Cursor(cfg.screen_inset_px)
    smoother = OneEuro2D(cfg.smooth_min_cutoff, cfg.smooth_beta)
    dwell = DwellClicker(cfg) if click_mode == "dwell" else None
    blink = BlinkClicker(cfg) if click_mode == "blink" else None
    paused = False

    try:
        while True:
            frame, s = tracker.read()
            if frame is None:
                continue

            status = "PAUSED (space to resume)" if paused else "LIVE"
            if s.ok and not paused:
                nx, ny = cmap.predict(features(s, cmap.mode))
                nx, ny = smoother.apply(nx, ny, time.monotonic())
                cursor.move_norm(nx, ny)
                if dwell is not None:
                    px, py = cursor.position()
                    if dwell.update(px, py):
                        cursor.click()
                if blink is not None and blink.update(s.ear):
                    cursor.click()

            lines = [f"{status} - q quit, space pause"]
            if not s.ok:
                lines.append("NO FACE")
            elif dwell is not None and dwell.progress() > 0:
                lines.append("dwell " + "#" * int(dwell.progress() * 10))
            view = _draw_hud(frame, s, lines)
            cv2.imshow("touchless", cv2.resize(view, (view.shape[1] // 2, view.shape[0] // 2)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                smoother.reset()
    except pyautogui.FailSafeException:
        print("Fail-safe triggered (mouse in top-left corner). Stopped.")
    finally:
        tracker.close()
        cv2.destroyAllWindows()
