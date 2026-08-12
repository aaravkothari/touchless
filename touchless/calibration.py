"""Calibration: learn the mapping from face features -> screen position.

Webcam gaze has no idea where your screen is, how far you sit, or how your
eyes are shaped. So we ask you to look at a grid of dots and fit a small
regularized polynomial regression from feature vectors to normalized screen
coordinates. Head mode works the same way, just with different features.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .tracking import FaceSample, FaceTracker

MODES = ("gaze", "head")


def features(sample: FaceSample, mode: str) -> np.ndarray:
    """The feature vector the screen mapping is learned over.

    Gaze mode includes head pose too — iris offsets alone are ambiguous
    because turning your head also shifts the iris in its socket.
    """
    if mode == "gaze":
        return np.array([sample.gaze_x, sample.gaze_y, sample.yaw, sample.pitch])
    return np.array([sample.yaw, sample.pitch])


def _poly2(f: np.ndarray) -> np.ndarray:
    """Degree-2 polynomial expansion with bias: [1, f_i, f_i*f_j]."""
    terms = [1.0, *f]
    n = len(f)
    for i in range(n):
        for j in range(i, n):
            terms.append(f[i] * f[j])
    return np.array(terms)


@dataclass
class CalibrationMap:
    mode: str
    weights: np.ndarray  # (n_terms, 2)

    @classmethod
    def fit(cls, feats: list[np.ndarray], targets: list[tuple[float, float]],
            mode: str, ridge_lambda: float) -> "CalibrationMap":
        X = np.stack([_poly2(f) for f in feats])          # (n, t)
        Y = np.array(targets, dtype=np.float64)           # (n, 2)
        # Ridge regression: (X'X + lambda*I)^-1 X'Y. The regularization keeps
        # the quadratic terms tame with only ~9 calibration points.
        t = X.shape[1]
        W = np.linalg.solve(X.T @ X + ridge_lambda * np.eye(t), X.T @ Y)
        return cls(mode=mode, weights=W)

    def predict(self, f: np.ndarray) -> tuple[float, float]:
        """Feature vector -> normalized screen coords (may exceed [0,1]; clamp later)."""
        out = _poly2(f) @ self.weights
        return float(out[0]), float(out[1])

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump({"mode": self.mode, "weights": self.weights.tolist()}, fh)

    @classmethod
    def load(cls, path: str) -> "CalibrationMap":
        with open(path) as fh:
            data = json.load(fh)
        return cls(mode=data["mode"], weights=np.array(data["weights"]))


def _grid_targets(cfg: Config) -> list[tuple[float, float]]:
    """Calibration dot positions in normalized [0,1] screen coordinates."""
    m, n = cfg.calib_margin, cfg.calib_grid
    axis = np.linspace(m, 1.0 - m, n)
    return [(float(x), float(y)) for y in axis for x in axis]


def run_calibration(cfg: Config, mode: str, screen_w: int, screen_h: int) -> CalibrationMap | None:
    """Fullscreen dot sequence. Returns the fitted map, or None if aborted (ESC)."""
    assert mode in MODES
    tracker = FaceTracker(cfg)
    win = "touchless calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    targets = _grid_targets(cfg)
    collected_feats: list[np.ndarray] = []
    collected_targets: list[tuple[float, float]] = []

    try:
        for i, (tx, ty) in enumerate(targets):
            px, py = int(tx * screen_w), int(ty * screen_h)
            phase_start = time.monotonic()
            samples: list[np.ndarray] = []
            while True:
                elapsed = time.monotonic() - phase_start
                capturing = elapsed >= cfg.calib_settle_s
                done = elapsed >= cfg.calib_settle_s + cfg.calib_capture_s

                frame, sample = tracker.read()
                if capturing and sample.ok:
                    samples.append(features(sample, mode))

                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                color = (0, 220, 0) if capturing else (0, 160, 255)
                cv2.circle(canvas, (px, py), 14, color, -1)
                cv2.circle(canvas, (px, py), 26, color, 2)
                cv2.putText(canvas, f"Look at the dot  ({i + 1}/{len(targets)})",
                            (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                if not sample.ok:
                    cv2.putText(canvas, "NO FACE DETECTED", (40, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    return None
                if done:
                    break

            if not samples:
                # Face was never detected for this dot; abort rather than fit garbage.
                print("No face samples collected for a dot - aborting calibration.")
                return None
            collected_feats.append(np.median(np.stack(samples), axis=0))
            collected_targets.append((tx, ty))
    finally:
        tracker.close()
        cv2.destroyWindow(win)

    return CalibrationMap.fit(collected_feats, collected_targets, mode, cfg.ridge_lambda)
