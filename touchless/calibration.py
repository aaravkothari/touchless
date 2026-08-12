"""Calibration: learn the mapping from face features -> screen position.

Webcam gaze has no idea where your screen is, how far you sit, or how your
eyes are shaped, so we learn a per-user mapping by having you look at a grid
of dots. Design notes (each of these fixed a real accuracy bug):

  - Features are z-scored before regression. Raw features mix scales
    (blendshapes ~0.3, yaw ~30 deg, translation ~cm); one ridge penalty
    across unscaled features crushes the small-scale signals.
  - Expansion is linear + squared terms only (13 terms for 6 features).
    A full degree-2 expansion has more terms than calibration points and
    extrapolates wildly between dots.
  - Ridge lambda is picked by leave-one-out cross-validation, not hardcoded.
  - Per-dot samples go through MAD outlier rejection (blinks and glances
    away during capture would otherwise poison the target).
  - A validation pass measures real error on 5 held-out dots and shows it
    to you before anything is saved.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .tracking import FaceSample, FaceTracker

CALIB_VERSION = 2

# Standardized features are clipped here before expansion: quadratic terms
# explode when the input drifts far outside the calibrated range.
Z_CLIP = 3.0


def _expand(z: np.ndarray) -> np.ndarray:
    """Standardized features -> [1, z, z^2] (no cross terms, stays determined)."""
    return np.concatenate(([1.0], z, z * z))


def _ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression that doesn't penalize the bias column."""
    t = X.shape[1]
    P = np.eye(t)
    P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * P, X.T @ Y)


def robust_mean(samples: np.ndarray, mad_z: float) -> np.ndarray:
    """Mean of rows after MAD outlier rejection (falls back to median)."""
    med = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - med), axis=0)
    z = np.abs(samples - med) / (1.4826 * mad + 1e-9)
    keep = (z < mad_z).all(axis=1)
    return samples[keep].mean(axis=0) if keep.any() else med


@dataclass
class CalibrationMap:
    mean: np.ndarray     # (6,) feature standardization
    std: np.ndarray      # (6,)
    weights: np.ndarray  # (13, 2)

    @classmethod
    def fit(cls, feats: list[np.ndarray], targets: list[tuple[float, float]],
            lambdas: tuple[float, ...]) -> "CalibrationMap":
        F = np.stack(feats)
        Y = np.array(targets, dtype=np.float64)
        mean = F.mean(axis=0)
        std = F.std(axis=0) + 1e-9
        X = np.stack([_expand(z) for z in (F - mean) / std])

        # Leave-one-out CV: n is tiny (16), brute force is instant.
        n = X.shape[0]
        best_lam, best_err = lambdas[0], np.inf
        for lam in lambdas:
            err = 0.0
            for i in range(n):
                idx = np.arange(n) != i
                W = _ridge(X[idx], Y[idx], lam)
                err += float(np.sum((X[i] @ W - Y[i]) ** 2))
            if err < best_err:
                best_lam, best_err = lam, err
        print(f"calibration: ridge lambda={best_lam} "
              f"(LOOCV rmse={np.sqrt(best_err / n):.4f} of screen)")
        return cls(mean=mean, std=std, weights=_ridge(X, Y, best_lam))

    def predict(self, f: np.ndarray) -> tuple[float, float]:
        """Feature vector -> normalized screen coords (may exceed [0,1]; clamp later)."""
        z = np.clip((f - self.mean) / self.std, -Z_CLIP, Z_CLIP)
        out = _expand(z) @ self.weights
        return float(out[0]), float(out[1])

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump({
                "version": CALIB_VERSION,
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
                "weights": self.weights.tolist(),
            }, fh)

    @classmethod
    def load(cls, path: str) -> "CalibrationMap":
        with open(path) as fh:
            data = json.load(fh)
        if data.get("version") != CALIB_VERSION:
            raise ValueError(
                f"{path} is from an incompatible version - recalibrate:\n"
                "  python -m touchless calibrate"
            )
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]),
                   weights=np.array(data["weights"]))


def _grid_targets(cfg: Config) -> list[tuple[float, float]]:
    """Calibration dot positions in normalized [0,1] screen coordinates."""
    m, n = cfg.calib_margin, cfg.calib_grid
    axis = np.linspace(m, 1.0 - m, n)
    return [(float(x), float(y)) for y in axis for x in axis]


def _check_targets(cfg: Config) -> list[tuple[float, float]]:
    """Validation dots: corners + center, offset from the calibration grid."""
    m = cfg.calib_margin + 0.07
    return [(m, m), (1 - m, m), (0.5, 0.5), (m, 1 - m), (1 - m, 1 - m)]


def _collect_dot(tracker: FaceTracker, cfg: Config, win: str, canvas_wh: tuple[int, int],
                 target: tuple[float, float], label: str,
                 predict=None) -> np.ndarray | None:
    """Show one dot, collect features. Returns robust feature mean, or None on ESC/no-face."""
    w, h = canvas_wh
    px, py = int(target[0] * w), int(target[1] * h)
    start = time.monotonic()
    samples: list[np.ndarray] = []
    while True:
        elapsed = time.monotonic() - start
        capturing = elapsed >= cfg.calib_settle_s
        done = elapsed >= cfg.calib_settle_s + cfg.calib_capture_s

        frame, sample = tracker.read()
        if capturing and sample.ok:
            samples.append(sample.features)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        color = (0, 220, 0) if capturing else (0, 160, 255)
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
        if done:
            break
    if not samples:
        print("No face samples collected for a dot - aborting.")
        return None
    return robust_mean(np.stack(samples), cfg.calib_mad_z)


def run_calibration(cfg: Config, screen_w: int, screen_h: int) -> CalibrationMap | None:
    """Collect 16 dots, fit, validate on 5 more, let the user accept or redo.

    Returns the fitted map, or None if aborted (ESC).
    """
    tracker = FaceTracker(cfg)
    win = "touchless calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    wh = (screen_w, screen_h)

    try:
        while True:  # collect -> fit -> validate; loop again if user hits R
            targets = _grid_targets(cfg)
            feats: list[np.ndarray] = []
            for i, t in enumerate(targets):
                f = _collect_dot(tracker, cfg, win, wh, t,
                                 f"Look at the dot  ({i + 1}/{len(targets)})")
                if f is None:
                    return None
                feats.append(f)
            cmap = CalibrationMap.fit(feats, targets, cfg.ridge_lambdas)

            # Validation on dots the fit has never seen.
            checks = _check_targets(cfg)
            errors_px: list[float] = []
            for i, t in enumerate(checks):
                f = _collect_dot(tracker, cfg, win, wh, t,
                                 f"Check: look at the dot  ({i + 1}/{len(checks)})",
                                 predict=cmap.predict)
                if f is None:
                    return None
                nx, ny = cmap.predict(f)
                errors_px.append(float(np.hypot((nx - t[0]) * screen_w,
                                                (ny - t[1]) * screen_h)))

            mean_err, max_err = np.mean(errors_px), np.max(errors_px)
            print(f"validation: mean {mean_err:.0f}px, max {max_err:.0f}px "
                  f"(screen {screen_w}x{screen_h})")
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            ok_quality = mean_err < 0.08 * screen_w
            for n, line in enumerate([
                f"Accuracy: mean {mean_err:.0f}px, worst {max_err:.0f}px",
                "(a decent result is ~5% of screen width, "
                f"= {0.05 * screen_w:.0f}px here)",
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
                    return cmap
                if key in (ord("r"), ord("R")):
                    break             # outer loop restarts collection
                if key == 27:         # ESC
                    return None
    finally:
        tracker.close()
        cv2.destroyWindow(win)
