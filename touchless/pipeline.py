"""Hand-mode pointer -> cursor pipeline, extracted for testability.

The transform from a raw fingertip position to a cursor move has grown
several interacting stages (stillness gate, anchor, median pre-filter,
One Euro, deadzone). Keeping it in one class lets the exact same code run
in the live app and in offline simulations (scripts/sim_stability.py),
so stability tuning doesn't require a camera and a steady hand.
"""

from collections import deque

import numpy as np

from .config import Config
from .smoothing import OneEuro2D


class HandPointerPipeline:
    """pointer (camera / hand-size units) -> normalized screen target.

    Stages: stillness gate (anchor absorption) -> anchor offset * gain ->
    clamp -> median-of-3 -> One Euro -> deadzone with hysteresis.
    """

    def __init__(self, cfg: Config, wrist: bool, screen_w: int, screen_h: int):
        self.cfg = cfg
        self.gain_x = cfg.hand_wrist_gain_x if wrist else cfg.hand_gain_x
        self.gain_y = cfg.hand_wrist_gain_y if wrist else cfg.hand_gain_y
        if wrist:
            self.smoother = OneEuro2D(cfg.hand_wrist_smooth_min_cutoff,
                                      cfg.hand_wrist_smooth_beta)
        else:
            self.smoother = OneEuro2D(cfg.hand_smooth_min_cutoff,
                                      cfg.hand_smooth_beta)
        self.dead_x = cfg.hand_deadzone_px / screen_w
        self.dead_y = cfg.hand_deadzone_px / screen_h
        self.sent: tuple[float, float] | None = None  # last position forwarded
        self.raw_hist: deque[tuple[float, float]] = deque(maxlen=3)  # spike pre-filter
        self.anchor: np.ndarray | None = None
        # Stillness gate state: while the finger isn't deliberately moving,
        # the anchor tracks the pointer 1:1 so landmark drift never reaches
        # the cursor (drift is unbounded, so no deadzone could absorb it).
        self.prev_pointer: np.ndarray | None = None
        self.win: deque[tuple[float, np.ndarray]] = deque()  # (t, gain-scaled pointer)
        self.still = False
        self.lock: np.ndarray | None = None  # gain-scaled position we locked at
        self.spread = 9.9
        self._gp: np.ndarray | None = None   # last gain-scaled pointer (HUD)

    def update(self, pointer: np.ndarray, now: float,
               recenter: bool = False) -> tuple[float, float, bool]:
        """One frame. Returns (sx, sy, moved): the smoothed normalized
        target, and whether the deadzone let it through (i.e. the OS cursor
        should be repositioned). recenter=True applies the tongue-recenter
        (re-anchor + filter reset) at the same point in the stage sequence
        as the original inline code."""
        if self.anchor is None:
            self.anchor = pointer.copy()  # first sighting = neutral

        # Stillness gate, positional: jitter makes instantaneous velocity
        # useless (wrist mode never read as still), but jitter clusters
        # around a point - so judge stillness by the spread of recent
        # gain-scaled positions instead.
        gp = np.array([self.gain_x * pointer[0], self.gain_y * pointer[1]])
        self._gp = gp
        self.win.append((now, gp))
        while self.win and now - self.win[0][0] > self.cfg.hand_still_window_s:
            self.win.popleft()
        if self.still:
            # Exit test FIRST: the frame that breaks the lock must NOT be
            # folded into the anchor. Absorb-then-unlock bakes the breaking
            # delta (often a one-frame landmark spike) into the anchor
            # permanently - the spike reverts next frame but the anchor
            # doesn't, teleporting the cursor by gain*spike on every
            # spurious unlock.
            if float(np.linalg.norm(gp - self.lock)) > self.cfg.hand_still_exit:
                self.still = False
                self.win.clear()  # relocking needs a fresh quiet window
            else:
                if self.prev_pointer is not None:
                    # Fold landmark wander into the anchor: d stays
                    # constant, cursor rock-solid, and when movement
                    # resumes there's no jump and no built-up drift.
                    self.anchor += pointer - self.prev_pointer
                # The lock point creeps after slow drift so drift alone
                # never unlocks; deliberate moves outrun it and do.
                self.lock += self.cfg.hand_still_lock_adapt * (gp - self.lock)
        elif (len(self.win) >= 3
              and now - self.win[0][0] >= 0.8 * self.cfg.hand_still_window_s):
            pts = np.stack([p for _, p in self.win])
            self.spread = float(np.max(
                np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
            if self.spread < self.cfg.hand_still_enter:
                self.still = True
                self.lock = gp.copy()
        self.prev_pointer = pointer.copy()

        if recenter:
            self.anchor = pointer.copy()
            self.smoother.reset()
            self.raw_hist.clear()
            self.sent = (0.5, 0.5)

        d = pointer - self.anchor
        nx = 0.5 - self.gain_x * float(d[0])  # x flipped: mirror-natural
        ny = 0.5 + self.gain_y * float(d[1])
        nx = float(np.clip(nx, -self.cfg.pred_clamp, 1 + self.cfg.pred_clamp))
        ny = float(np.clip(ny, -self.cfg.pred_clamp, 1 + self.cfg.pred_clamp))
        self.raw_hist.append((nx, ny))
        mx = float(np.median([p[0] for p in self.raw_hist]))
        my = float(np.median([p[1] for p in self.raw_hist]))
        sx, sy = self.smoother.apply(mx, my, now)
        # Deadzone with hysteresis: sub-pixel-scale wobble around the last
        # sent position is swallowed; a slow drift accumulates until it
        # clears the threshold and then goes through.
        moved = (self.sent is None or abs(sx - self.sent[0]) > self.dead_x
                 or abs(sy - self.sent[1]) > self.dead_y)
        if moved:
            self.sent = (sx, sy)
        return sx, sy, moved

    def pointer_lost(self):
        """Pointer left the frame: hold position, forget stillness state."""
        self.prev_pointer = None
        self.still = False
        self.win.clear()

    def pause_reset(self):
        """Pause toggled: drop all filter state so resume starts clean."""
        self.smoother.reset()
        self.raw_hist.clear()
        self.sent = None
        self.prev_pointer = None
        self.still = False
        self.win.clear()

    def still_info(self) -> str:
        """[STILL] HUD line for threshold tuning."""
        if self.still and self.lock is not None and self._gp is not None:
            gap = float(np.linalg.norm(self._gp - self.lock))
            return (f"still LOCKED   gap {gap:.3f} "
                    f"(> {self.cfg.hand_still_exit:.3f} unlocks)")
        return (f"still live     spread {self.spread:.3f} "
                f"(< {self.cfg.hand_still_enter:.3f} locks)")
