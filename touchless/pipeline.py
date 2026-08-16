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
        self._absorbed: np.ndarray | None = None  # pointer at last absorbed frame
        self.win: deque[tuple[float, np.ndarray]] = deque()  # (t, gain-scaled pointer)
        self.still = False
        self.lock: np.ndarray | None = None  # gain-scaled position we locked at
        self.spread = 9.9
        self._gate_hist: deque[np.ndarray] = deque(maxlen=3)  # gate spike filter
        self._gp: np.ndarray | None = None   # last gate signal (HUD)
        self._exit_pending = 0  # consecutive frames past the exit threshold
        # Online noise-scale estimate: recent |delta gate signal|, read at
        # the 25th percentile so movement periods can't inflate it.
        self._deltas: deque[float] = deque(maxlen=60)
        self._prev_ggp: np.ndarray | None = None
        self._enter_eff = cfg.hand_still_enter  # effective thresholds (HUD)
        self._exit_eff = cfg.hand_still_exit
        self._creep_acc = 0.0  # slow-creep escape: leaky time accumulator
        self._creep_t: float | None = None
        # Absorption history for the creep rollback: (t, pointer) at each
        # absorbed frame, kept a bit longer than the creep timer.
        self._absorb_hist: deque[tuple[float, np.ndarray]] = deque()

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
        # The gate judges the median-of-3 of the gain-scaled pointer, not
        # the raw value: a single spike frame would otherwise blow the
        # window spread past the enter threshold (gate never locks) or the
        # exit threshold (gate flaps), and spike frames are exactly the
        # noise the gate exists to absorb.
        self._gate_hist.append(gp)
        ggp = np.median(np.stack(self._gate_hist), axis=0)
        self._gp = ggp
        self.win.append((now, ggp))
        while self.win and now - self.win[0][0] > self.cfg.hand_still_horizon_s:
            self.win.popleft()
        if self._prev_ggp is not None:
            self._deltas.append(float(np.linalg.norm(ggp - self._prev_ggp)))
        self._prev_ggp = ggp
        if len(self._deltas) >= 20:
            # Trimmed mean (25-75%) of the deltas, not a low percentile:
            # the median pre-filter makes many consecutive gate values
            # identical, so a low percentile sits on the zero-cluster
            # boundary and oscillates ~3x window to window, alternately
            # over-tightening the exit threshold (spurious unlocks).
            d = np.fromiter(self._deltas, float)
            lo, hi = np.percentile(d, [25, 75])
            mid = d[(d >= lo) & (d <= hi)]
            nd = float(mid.mean()) if len(mid) else 0.0
            self._enter_eff = max(self.cfg.hand_still_enter,
                                  self.cfg.hand_still_noise_enter * nd)
            self._exit_eff = max(self.cfg.hand_still_exit,
                                 self.cfg.hand_still_noise_exit * nd)
        if self.still:
            # Exit test FIRST: the frame that breaks the lock must NOT be
            # folded into the anchor. Absorb-then-unlock bakes the breaking
            # delta (often a one-frame landmark spike) into the anchor
            # permanently - the spike reverts next frame but the anchor
            # doesn't, teleporting the cursor by gain*spike on every
            # spurious unlock.
            gap = float(np.linalg.norm(ggp - self.lock))
            # Slow-creep escape: a deliberate move slower than the lock
            # EMA never reaches the exit threshold (the lock chases it,
            # equilibrium gap = speed/adapt) - accumulating time past half
            # the threshold unlocks anyway, so slow precise moves aren't
            # absorbed forever. Leaky accumulator, not a hard timer: the
            # gap rides the band with noise on top, and a hard timer
            # resets on every one-frame dip and never fires.
            dt = now - self._creep_t if self._creep_t is not None else 0.0
            self._creep_t = now
            if gap > 0.4 * self._exit_eff:
                self._creep_acc += dt
            else:
                self._creep_acc = max(0.0, self._creep_acc - dt)
            creeping = self._creep_acc >= self.cfg.hand_still_creep_s
            if creeping:
                self.still = False
                self._exit_pending = 0
                self._creep_acc = 0.0
                self._creep_t = None
                self.win.clear()
                # Roll back the creep episode's absorbed motion (bounded
                # to the last ~creep_s of history, so a long-parked drift
                # can never fling the cursor) and hold off relocking: one
                # spread-window can't tell this slow a move from
                # stillness, so without the holdoff the gate would relock
                # immediately and re-absorb the move forever.
                cut = now - self.cfg.hand_still_creep_s
                start = None
                for ht, hp in self._absorb_hist:
                    start = hp
                    if ht >= cut:
                        break
                if start is not None and self._absorbed is not None:
                    self.anchor -= self._absorbed - start
                self._absorb_hist.clear()
            elif gap > self._exit_eff:
                # Excursion frames are NOT absorbed into the anchor, so d
                # starts tracking them immediately: if the excursion recedes
                # (noise burst) d snaps back with zero residue; if it
                # persists (a real move) no motion was lost when the gate
                # finally opens.
                self._exit_pending += 1
                if self._exit_pending >= self.cfg.hand_still_exit_frames:
                    self.still = False
                    self._exit_pending = 0
                    self.win.clear()  # relocking needs a fresh quiet window
            else:
                self._exit_pending = 0
                # Fold landmark wander into the anchor: d stays constant,
                # cursor rock-solid, and when movement resumes there's no
                # jump and no built-up drift. Absorption spans from the
                # LAST ABSORBED frame, not the previous frame: a pending
                # excursion that recedes gets its whole net delta folded
                # in at once, so d snaps back to the lock-time value with
                # zero residue (per-frame deltas would leak the episode's
                # net noise offset into d permanently, a random walk).
                if self._absorbed is not None:
                    self.anchor += pointer - self._absorbed
                self._absorbed = pointer.copy()
                self._absorb_hist.append((now, pointer.copy()))
                cut = now - self.cfg.hand_still_creep_s - 0.5
                while self._absorb_hist and self._absorb_hist[0][0] < cut:
                    self._absorb_hist.popleft()
                # The lock point creeps after slow drift so drift alone
                # never unlocks; deliberate moves outrun it and do.
                self.lock += self.cfg.hand_still_lock_adapt * (ggp - self.lock)
        elif (len(self.win) >= 3
              and now - self.win[0][0] >= 0.8 * self.cfg.hand_still_horizon_s):
            w = self.cfg.hand_still_window_s
            t0 = self.win[0][0]
            recent = np.stack([p for t, p in self.win if t >= now - w])
            self.spread = float(np.max(
                np.linalg.norm(recent - recent.mean(axis=0), axis=1)))
            # Slow-motion guard: spread over one short window can't see a
            # slow move (its per-window displacement is below the noise
            # floor), but the net displacement across the whole horizon
            # can. Without this the gate relocks mid-move and re-absorbs
            # the motion forever.
            first = np.stack([p for t, p in self.win if t <= t0 + w])
            net = float(np.linalg.norm(recent.mean(axis=0) - first.mean(axis=0)))
            if (self.spread < self._enter_eff
                    and net < self.cfg.hand_still_net_mult * self._enter_eff):
                self.still = True
                # Lock onto the cluster center, not whatever sample
                # happened to arrive last (that can sit at the cluster
                # edge, leaving half the exit margin already spent).
                self.lock = recent.mean(axis=0)
                self._exit_pending = 0
                self._creep_acc = 0.0
                self._creep_t = None
                self._absorbed = pointer.copy()
                self._absorb_hist.clear()
                self._absorb_hist.append((now, pointer.copy()))

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
        self._absorbed = None
        self.still = False
        self._exit_pending = 0
        self._creep_acc = 0.0
        self._creep_t = None
        self._absorb_hist.clear()
        self.win.clear()
        self._prev_ggp = None  # a delta across the gap isn't a noise sample

    def pause_reset(self):
        """Pause toggled: drop all filter state so resume starts clean."""
        self.smoother.reset()
        self.raw_hist.clear()
        self.sent = None
        self._absorbed = None
        self.still = False
        self._exit_pending = 0
        self._creep_acc = 0.0
        self._creep_t = None
        self._absorb_hist.clear()
        self.win.clear()
        self._prev_ggp = None

    def still_info(self) -> str:
        """[STILL] HUD line for threshold tuning."""
        if self.still and self.lock is not None and self._gp is not None:
            gap = float(np.linalg.norm(self._gp - self.lock))
            return (f"still LOCKED   gap {gap:.3f} "
                    f"(> {self._exit_eff:.3f} unlocks)")
        return (f"still live     spread {self.spread:.3f} "
                f"(< {self._enter_eff:.3f} locks)")
