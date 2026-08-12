"""Calibration flow: pursuit collection -> model fit -> validation dots.

The mapping itself is learned in model.py (candidates compete on held-out
data); collection happens in pursuit.py (follow the moving cursor). This
module owns the UX around them: fullscreen window, instruction screens,
the arrival-gated static validation dots, and the accept/redo loop.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from .config import Config
from .model import GazeModel
from .pursuit import collect
from .tracking import FaceTracker


def robust_mean(samples: np.ndarray, mad_z: float) -> np.ndarray:
    """Mean of rows after MAD outlier rejection (falls back to median)."""
    med = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - med), axis=0)
    z = np.abs(samples - med) / (1.4826 * mad + 1e-9)
    keep = (z < mad_z).all(axis=1)
    return samples[keep].mean(axis=0) if keep.any() else med


def _check_targets(cfg: Config) -> list[tuple[float, float]]:
    """Validation dots: corners + center."""
    m = cfg.calib_margin + 0.10
    return [(m, m), (1 - m, m), (0.5, 0.5), (m, 1 - m), (1 - m, 1 - m)]


def _instruction_screen(win: str, wh: tuple[int, int], lines: list[str]) -> bool:
    """Fullscreen instructions; any key continues, ESC aborts. True = continue."""
    w, h = wh
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for n, line in enumerate(lines):
        cv2.putText(canvas, line, (60, 140 + n * 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (200, 200, 200), 2)
    cv2.putText(canvas, "press any key to continue  (ESC aborts)",
                (60, 140 + len(lines) * 60 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 160, 255), 2)
    cv2.imshow(win, canvas)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            return False
        if key != 255:
            return True


def _collect_dot(tracker: FaceTracker, cfg: Config, win: str, wh: tuple[int, int],
                 target: tuple[float, float], prev: tuple[float, float],
                 label: str, predict=None) -> np.ndarray | None:
    """Glide the dot from prev to target, wait for the gaze to land, capture.

    Returns robust feature mean, or None on ESC / persistent no-face.
    """
    w, h = wh
    start = time.monotonic()
    gaze_win: deque[tuple[float, float, float]] = deque()  # (t, gaze_x, gaze_y)
    samples: list[np.ndarray] = []
    capture_started: float | None = None
    warned = False

    while True:
        now = time.monotonic()
        elapsed = now - start

        frame, sample = tracker.read()

        # Dot position: ease from prev to target during the glide phase.
        a = min(elapsed / cfg.calib_glide_s, 1.0)
        a = a * a * (3 - 2 * a)  # smoothstep
        dx = prev[0] + (target[0] - prev[0]) * a
        dy = prev[1] + (target[1] - prev[1]) * a
        px, py = int(dx * w), int(dy * h)
        glide_done = elapsed >= cfg.calib_glide_s

        if glide_done and sample.ok and sample.blink < cfg.blink_gate:
            gaze_win.append((now, sample.gaze_x, sample.gaze_y))
            while gaze_win and now - gaze_win[0][0] > 0.4:
                gaze_win.popleft()
            if capture_started is None and len(gaze_win) >= 5:
                g = np.array([(gx, gy) for _, gx, gy in gaze_win])
                if float(g.std(axis=0).max()) < cfg.calib_stability_std:
                    capture_started = now  # gaze has landed
            if capture_started is not None:
                samples.append(sample.features)

        # Timeout: capture anyway rather than stalling the whole sequence.
        if capture_started is None and glide_done \
                and elapsed > cfg.calib_glide_s + cfg.calib_timeout_s:
            if not warned:
                print(f"note: gaze never stabilized on dot at {target}, "
                      "capturing anyway")
                warned = True
            capture_started = now

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        color = (0, 220, 0) if capture_started is not None else (0, 160, 255)
        cv2.circle(canvas, (px, py), 14, color, -1)
        cv2.circle(canvas, (px, py), 26, color, 2)
        cv2.putText(canvas, label, (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (200, 200, 200), 2)
        if not sample.ok:
            cv2.putText(canvas, "NO FACE DETECTED", (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if predict is not None and sample.ok:
            nx, ny = predict(sample.features)
            qx = int(np.clip(nx, 0, 1) * (w - 1))
            qy = int(np.clip(ny, 0, 1) * (h - 1))
            cv2.drawMarker(canvas, (qx, qy), (255, 0, 255),
                           cv2.MARKER_CROSS, 30, 2)
        cv2.imshow(win, canvas)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            return None
        if capture_started is not None \
                and now - capture_started >= cfg.calib_capture_s:
            break

    if not samples:
        print("No usable samples for a dot - aborting.")
        return None
    return robust_mean(np.stack(samples), cfg.calib_mad_z)


def run_calibration(cfg: Config, screen_w: int, screen_h: int) -> GazeModel | None:
    """Pursuit collect -> fit -> validate -> user accepts or redoes.

    Returns the fitted model (already backed by gaze_data.npz on disk),
    or None if aborted (ESC).
    """
    tracker = FaceTracker(cfg)
    win = "touchless calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    wh = (screen_w, screen_h)

    try:
        while True:  # collect -> fit -> validate; loop again if user hits R
            data = collect(tracker, cfg, win, screen_w, screen_h,
                           _instruction_screen)
            if data is None:
                return None
            data.save(cfg.data_file)
            print(f"pursuit session saved to {cfg.data_file} "
                  "(retrain offline with: python -m touchless retrain)")
            model = GazeModel.fit(data, cfg, screen_w, screen_h)

            # Validation on static dots the model has never seen: 5 at
            # normal posture, 1 leaned back.
            if not _instruction_screen(win, wh, ["Check: sit NORMALLY."]):
                return None
            errors_px: list[float] = []
            prev = (0.5, 0.5)
            for i, t in enumerate(_check_targets(cfg)):
                f = _collect_dot(tracker, cfg, win, wh, t, prev,
                                 f"Check  ({i + 1}/5)", predict=model.predict)
                if f is None:
                    return None
                nx, ny = model.predict(f)
                errors_px.append(float(np.hypot((nx - t[0]) * screen_w,
                                                (ny - t[1]) * screen_h)))
                prev = t
            if not _instruction_screen(win, wh, ["Check: now LEAN BACK."]):
                return None
            f = _collect_dot(tracker, cfg, win, wh, (0.5, 0.5), (0.5, 0.5),
                             "Leaned-back check", predict=model.predict)
            if f is None:
                return None
            nx, ny = model.predict(f)
            lean_err = float(np.hypot((nx - 0.5) * screen_w, (ny - 0.5) * screen_h))

            mean_err, max_err = np.mean(errors_px), np.max(errors_px)
            print(f"validation: normal mean {mean_err:.0f}px, max {max_err:.0f}px, "
                  f"leaned-back {lean_err:.0f}px (screen {screen_w}x{screen_h})")
            ok_quality = mean_err < 0.08 * screen_w and lean_err < 0.12 * screen_w
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            for n, line in enumerate([
                f"Model: {model.name} (holdout {model.holdout_px:.0f}px, "
                f"lag {model.lag_ms:.0f}ms)",
                f"Normal posture: mean {mean_err:.0f}px, worst {max_err:.0f}px",
                f"Leaned back:    {lean_err:.0f}px",
                f"(a decent result is ~{0.05 * screen_w:.0f}px mean here)",
                "",
                "ENTER  save and finish" + ("" if ok_quality else "  (quality is poor)"),
                "R      redo calibration",
                "ESC    abort without saving",
            ]):
                cv2.putText(canvas, line, (60, 120 + n * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (200, 200, 200) if ok_quality else (0, 160, 255), 2)
            cv2.imshow(win, canvas)
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key in (13, 10):   # Enter
                    return model
                if key in (ord("r"), ord("R")):
                    break             # outer loop restarts collection
                if key == 27:         # ESC
                    return None
    finally:
        tracker.close()
        cv2.destroyWindow(win)
