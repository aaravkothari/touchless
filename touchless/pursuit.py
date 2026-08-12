"""Pursuit calibration data collection: follow the moving cursor with your eyes.

Instead of staring at a handful of static dots, you track a smoothly
wandering target for ~2 minutes while the app records (face features ->
target position) pairs every frame. That yields thousands of training
samples covering the whole screen — enough to train a real ML model —
and phase 2 has you changing posture *while* tracking, so posture
invariance is learned from continuous data rather than assumed.

The real OS cursor is moved along with the rendered dot, so you are
literally watching the mouse the way you will during use.
"""

from __future__ import annotations

from dataclasses import dataclass

import time

import cv2
import numpy as np
import pyautogui

from .config import Config
from .tracking import FaceTracker


@dataclass
class PursuitData:
    t: np.ndarray        # (n,) seconds, monotonic
    X: np.ndarray        # (n, 8) features
    blink: np.ndarray    # (n,)
    target: np.ndarray   # (n, 2) normalized screen position of the dot
    phase: np.ndarray    # (n,) which collection phase each row came from

    def save(self, path: str):
        np.savez(path, t=self.t, X=self.X, blink=self.blink,
                 target=self.target, phase=self.phase)

    @classmethod
    def load(cls, path: str) -> "PursuitData":
        d = np.load(path)
        return cls(t=d["t"], X=d["X"], blink=d["blink"],
                   target=d["target"], phase=d["phase"])

    @classmethod
    def concat(cls, parts: list["PursuitData"]) -> "PursuitData":
        return cls(*(np.concatenate([getattr(p, f) for p in parts])
                     for f in ("t", "X", "blink", "target", "phase")))


class _Wander:
    """Smooth random trajectory: eased travel between waypoints, random holds.

    Holds give fixation-quality samples; travel gives coverage. Every Nth
    waypoint is pushed to an edge/corner so the borders get trained too.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.pos = np.array([0.5, 0.5])
        self.src = self.pos.copy()
        self.dst = self.pos.copy()
        self.seg_start = 0.0
        self.seg_dur = 1e-9
        self.holding = False
        self.n_waypoints = 0

    def _next_waypoint(self) -> np.ndarray:
        m = self.cfg.calib_margin
        p = self.rng.uniform(m, 1 - m, size=2)
        self.n_waypoints += 1
        if self.n_waypoints % self.cfg.pursuit_edge_every == 0:
            axis = self.rng.integers(2)
            p[axis] = m if self.rng.random() < 0.5 else 1 - m
        return p

    def pos_at(self, now: float) -> np.ndarray:
        a = (now - self.seg_start) / self.seg_dur
        if a >= 1.0:
            # Segment finished: either hold here or travel somewhere new.
            self.seg_start = now
            if not self.holding and self.rng.random() < self.cfg.pursuit_hold_prob:
                self.holding = True
                self.src = self.dst.copy()
                self.seg_dur = float(self.rng.uniform(*self.cfg.pursuit_hold_s))
            else:
                self.holding = False
                self.src = self.dst.copy()
                self.dst = self._next_waypoint()
                speed = float(self.rng.uniform(*self.cfg.pursuit_speed))
                dist = float(np.linalg.norm(self.dst - self.src))
                self.seg_dur = max(dist / speed, 1e-3)
            a = 0.0
        if self.holding:
            self.pos = self.dst.copy()
        else:
            a = min(max(a, 0.0), 1.0)
            a = a * a * (3 - 2 * a)  # smoothstep easing
            self.pos = self.src + (self.dst - self.src) * a
        return self.pos


_PHASES = [
    ["Phase 1 of 2: follow the moving cursor with your eyes.",
     "Sit comfortably. Just watch it - do not fight to keep your head still."],
    ["Phase 2 of 2: keep following the cursor...",
     "...and SLOWLY change posture while you do: lean back,",
     "lean in, shift left and right, sit tall, slouch."],
]


def collect(tracker: FaceTracker, cfg: Config, win: str,
            screen_w: int, screen_h: int,
            instruction_screen) -> PursuitData | None:
    """Run both pursuit phases. Returns collected data, or None on ESC."""
    parts: list[PursuitData] = []
    rng = np.random.default_rng()

    for phase_i, lines in enumerate(_PHASES):
        if not instruction_screen(win, (screen_w, screen_h), lines):
            return None
        wander = _Wander(cfg, rng)
        rows_t, rows_x, rows_b, rows_tg = [], [], [], []
        start = time.monotonic()
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= cfg.pursuit_phase_s:
                break

            p = wander.pos_at(now)
            px, py = int(p[0] * screen_w), int(p[1] * screen_h)
            try:
                pyautogui.moveTo(max(px, 2), max(py, 2))  # you watch the real mouse
            except pyautogui.FailSafeException:
                return None

            frame, sample = tracker.read()
            if sample.ok:
                rows_t.append(now)
                rows_x.append(sample.features)
                rows_b.append(sample.blink)
                rows_tg.append(p.copy())

            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(canvas, (px, py), 16, (0, 220, 0), -1)
            cv2.circle(canvas, (px, py), 30, (0, 220, 0), 2)
            frac = elapsed / cfg.pursuit_phase_s
            cv2.rectangle(canvas, (0, 0), (int(screen_w * frac), 8), (0, 160, 255), -1)
            if not sample.ok:
                cv2.putText(canvas, "NO FACE DETECTED", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                return None

        n = len(rows_t)
        if n < 100:
            print(f"phase {phase_i + 1}: only {n} usable frames - aborting "
                  "(face not tracked? check lighting with `preview`)")
            return None
        parts.append(PursuitData(
            t=np.array(rows_t), X=np.stack(rows_x), blink=np.array(rows_b),
            target=np.stack(rows_tg), phase=np.full(n, phase_i)))
        print(f"phase {phase_i + 1}: {n} samples")

    return PursuitData.concat(parts)
