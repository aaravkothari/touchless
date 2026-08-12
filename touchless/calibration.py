"""Calibration: learn the mapping from face features -> screen position.

Webcam gaze has no idea where your screen is, how far you sit, or how your
eyes are shaped, so we learn a per-user mapping by having you look at dots.
Design notes (each of these fixed a real accuracy bug):

  - The model has explicit tz-interaction terms. Physics: screen
    displacement ~ distance x tan(angle), so depth multiplies every angular
    signal. Without those terms, leaning breaks the mapping.
  - Calibration runs at THREE postures (normal / lean back / lean in).
    A single-posture fit has ~zero variance in head position, so the model
    cannot learn position invariance no matter its form.
  - Design-matrix columns are z-scored before ridge (mixed scales otherwise
    crush the small signals), lambda picked by leave-one-out CV, columns
    clipped at predict time so extrapolation can't explode the quadratics.
  - Capture per dot is gaze-arrival-gated: samples start only once your
    gaze has actually landed (rolling std low, no blink), instead of a
    fixed timer that captures your eyes mid-flight.
  - A validation pass measures real error on held-out dots - including one
    at a different posture - and shows it before anything is saved.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .tracking import FaceTracker

CALIB_VERSION = 3

# Standardized design-matrix columns are clipped here at predict time:
# quadratic/interaction terms explode when input drifts far outside the
# calibrated range.
Z_CLIP = 4.0

# Feature indices (see tracking.FEATURE_NAMES).
_GX, _GY, _YAW, _PITCH, _ROLL, _TX, _TY, _TZ = range(8)


def _expand(f: np.ndarray) -> np.ndarray:
    """Raw features -> 17 design terms.

    [1, 8 linear, tz*(gaze_x, gaze_y, yaw, pitch), (gaze_x, gaze_y, yaw, pitch)^2]
    The tz products encode "angular displacement scales with distance".
    """
    ang = f[[_GX, _GY, _YAW, _PITCH]]
    return np.concatenate(([1.0], f, f[_TZ] * ang, ang * ang))


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
    col_mean: np.ndarray  # (17,) design-column standardization (bias: 0)
    col_std: np.ndarray   # (17,)                               (bias: 1)
    weights: np.ndarray   # (17, 2)

    @classmethod
    def fit(cls, feats: list[np.ndarray], targets: list[tuple[float, float]],
            lambdas: tuple[float, ...]) -> "CalibrationMap":
        Xr = np.stack([_expand(f) for f in feats])
        Y = np.array(targets, dtype=np.float64)
        col_mean = Xr.mean(axis=0)
        col_std = Xr.std(axis=0) + 1e-9
        col_mean[0], col_std[0] = 0.0, 1.0  # keep the bias column as-is
        X = (Xr - col_mean) / col_std

        # Leave-one-out CV: n is small (~34), brute force is instant.
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
        return cls(col_mean=col_mean, col_std=col_std,
                   weights=_ridge(X, Y, best_lam))

    def predict(self, f: np.ndarray) -> tuple[float, float]:
        """Feature vector -> normalized screen coords (may exceed [0,1]; clamp later)."""
        x = (_expand(f) - self.col_mean) / self.col_std
        out = np.clip(x, -Z_CLIP, Z_CLIP) @ self.weights
        return float(out[0]), float(out[1])

    def save(self, path: str):
        with open(path, "w") as fh:
            json.dump({
                "version": CALIB_VERSION,
                "col_mean": self.col_mean.tolist(),
                "col_std": self.col_std.tolist(),
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
        return cls(col_mean=np.array(data["col_mean"]),
                   col_std=np.array(data["col_std"]),
                   weights=np.array(data["weights"]))


def _grid_targets(cfg: Config, n: int) -> list[tuple[float, float]]:
    """n x n dot positions in normalized [0,1] screen coordinates."""
    m = cfg.calib_margin
    axis = np.linspace(m, 1.0 - m, n)
    return [(float(x), float(y)) for y in axis for x in axis]


def _check_targets(cfg: Config) -> list[tuple[float, float]]:
    """Validation dots: corners + center, offset from the calibration grids."""
    m = cfg.calib_margin + 0.07
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


# (instruction lines, grid size) per posture stage.
_STAGES = [
    (["Stage 1 of 3: sit in your NORMAL position.",
      "Follow the dot with your eyes."], "calib_grid"),
    (["Stage 2 of 3: LEAN BACK in your chair.",
      "Stay leaned back for this whole stage."], "calib_stage_grid"),
    (["Stage 3 of 3: LEAN IN closer to the screen.",
      "Stay close for this whole stage."], "calib_stage_grid"),
]


def _run_stage_dots(tracker, cfg, win, wh, targets, stage_label,
                    predict=None) -> list[np.ndarray] | None:
    feats: list[np.ndarray] = []
    prev = (0.5, 0.5)
    for i, t in enumerate(targets):
        f = _collect_dot(tracker, cfg, win, wh, t, prev,
                         f"{stage_label}  ({i + 1}/{len(targets)})",
                         predict=predict)
        if f is None:
            return None
        feats.append(f)
        prev = t
    return feats


def run_calibration(cfg: Config, screen_w: int, screen_h: int) -> CalibrationMap | None:
    """3-posture collection -> fit -> validation -> user accepts or redoes.

    Returns the fitted map, or None if aborted (ESC).
    """
    tracker = FaceTracker(cfg)
    win = "touchless calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    wh = (screen_w, screen_h)

    try:
        while True:  # collect -> fit -> validate; loop again if user hits R
            all_feats: list[np.ndarray] = []
            all_targets: list[tuple[float, float]] = []
            for lines, grid_attr in _STAGES:
                if not _instruction_screen(win, wh, lines):
                    return None
                targets = _grid_targets(cfg, getattr(cfg, grid_attr))
                feats = _run_stage_dots(tracker, cfg, win, wh, targets,
                                        "Follow the dot")
                if feats is None:
                    return None
                all_feats += feats
                all_targets += targets
            cmap = CalibrationMap.fit(all_feats, all_targets, cfg.ridge_lambdas)

            # Validation on dots the fit has never seen: 5 at normal posture,
            # then 1 leaned back, so position-invariance is actually measured.
            if not _instruction_screen(win, wh, ["Check: sit NORMALLY again."]):
                return None
            errors_px: list[float] = []
            prev = (0.5, 0.5)
            for i, t in enumerate(_check_targets(cfg)):
                f = _collect_dot(tracker, cfg, win, wh, t, prev,
                                 f"Check  ({i + 1}/5)", predict=cmap.predict)
                if f is None:
                    return None
                nx, ny = cmap.predict(f)
                errors_px.append(float(np.hypot((nx - t[0]) * screen_w,
                                                (ny - t[1]) * screen_h)))
                prev = t
            if not _instruction_screen(win, wh, ["Check: now LEAN BACK."]):
                return None
            f = _collect_dot(tracker, cfg, win, wh, (0.5, 0.5), (0.5, 0.5),
                             "Leaned-back check", predict=cmap.predict)
            if f is None:
                return None
            nx, ny = cmap.predict(f)
            lean_err = float(np.hypot((nx - 0.5) * screen_w, (ny - 0.5) * screen_h))

            mean_err, max_err = np.mean(errors_px), np.max(errors_px)
            print(f"validation: normal mean {mean_err:.0f}px, max {max_err:.0f}px, "
                  f"leaned-back {lean_err:.0f}px (screen {screen_w}x{screen_h})")
            ok_quality = mean_err < 0.08 * screen_w and lean_err < 0.12 * screen_w
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            for n, line in enumerate([
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
                    return cmap
                if key in (ord("r"), ord("R")):
                    break             # outer loop restarts collection
                if key == 27:         # ESC
                    return None
    finally:
        tracker.close()
        cv2.destroyWindow(win)
