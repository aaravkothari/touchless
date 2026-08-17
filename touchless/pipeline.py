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
                                      cfg.hand_wrist_smooth_beta,
                                      cfg.hand_wrist_smooth_d_cutoff)
        else:
            self.smoother = OneEuro2D(cfg.hand_smooth_min_cutoff,
                                      cfg.hand_smooth_beta,
                                      cfg.hand_smooth_d_cutoff)
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
        # Online noise-scale estimate: recent |delta gate signal|, read as
        # a trimmed mean so neither movement periods (top of the range) nor
        # median-collision zeros (bottom) can skew it.
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
                # can never fling the cursor): the slow move lands where
                # the finger's cumulative motion says it should, smoothed
                # into a quick glide by the One Euro stage. The horizon
                # guard on relocking then keeps the move flowing.
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
                # Excursion frames are NOT absorbed into the anchor, and
                # the output stays frozen at the lock offset while they
                # pend: a receding noise burst leaves zero residue, an
                # absorbed re-detection step never blips the cursor, and
                # a real move's pent-up motion is delivered as a catch-up
                # glide when the gate opens - nothing is lost.
                self._exit_pending += 1
                hard_cap = 2 * self.cfg.hand_still_exit_frames
                if self._exit_pending >= self.cfg.hand_still_exit_frames:
                    # Re-detection check before unlocking: a palm
                    # re-detection re-solves the landmarks to a slightly
                    # different answer - a persistent step that would
                    # otherwise read as a real move and jump the cursor.
                    # Signature: the excursion is RESTING at its new
                    # position (last two gate values on top of each
                    # other) and the step is modest. A real move is still
                    # moving through this check and unlocks; a slow move
                    # that sneaks through once keeps moving and unlocks
                    # on the next 3-frame round, ~100 ms later.
                    tail = np.stack([p for _, p in list(self.win)[-2:]])
                    resting = (float(np.linalg.norm(tail[-1] - tail[0]))
                               < 0.7 * self._enter_eff)
                    step = float(np.linalg.norm(tail.mean(axis=0) - self.lock))
                    if (resting and step
                            < self.cfg.hand_still_step_mult * self._exit_eff):
                        if self._absorbed is not None:
                            self.anchor += pointer - self._absorbed
                        self._absorbed = pointer.copy()
                        self._absorb_hist.append((now, pointer.copy()))
                        self.lock = tail.mean(axis=0)
                        self._exit_pending = 0
                    elif self._exit_pending >= hard_cap:
                        # Still moving after the extended window: real.
                        self.still = False
                        self._exit_pending = 0
                        self.win.clear()  # relock needs a fresh quiet window
                    # else: ambiguous (noise blurred the resting test) -
                    # wait another frame. Cheap: the output is frozen at
                    # the lock offset while pending, and on a real unlock
                    # the pent-up motion is delivered as a catch-up glide,
                    # so the extra frames lose nothing.
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
            if self.still:
                self._absorbed = pointer.copy()  # keep frozen-d consistent
            self.smoother.reset()
            self.raw_hist.clear()
            self.sent = (0.5, 0.5)

        # While locked, the output offset is computed from the last
        # ABSORBED pointer, not the live one: in-sync frames are identical
        # (absorption just set them equal), and pending excursion frames
        # are held at the lock offset so a re-detection step under
        # evaluation can't blip the cursor.
        if self.still and self._absorbed is not None:
            d = self._absorbed - self.anchor
        else:
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
        self._gate_hist.clear()  # median across the gap would mix positions
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
        self._gate_hist.clear()
        self._prev_ggp = None

    def still_info(self) -> str:
        """[STILL] HUD line for threshold tuning."""
        if self.still and self.lock is not None and self._gp is not None:
            gap = float(np.linalg.norm(self._gp - self.lock))
            return (f"still LOCKED   gap {gap:.3f} "
                    f"(> {self._exit_eff:.3f} unlocks)")
        return (f"still live     spread {self.spread:.3f} "
                f"(< {self._enter_eff:.3f} locks)")


class ArmGate:
    """(tip, ref) -> virtual pointer that ignores whole-arm translation.

    Hand-pure mode's upstream stage: the virtual pointer tracks the
    absolute tip 1:1 while the arm is still (pure-x,y feel), but FREEZES
    while the palm-MCP centroid (ref) is translating - whole-arm motion,
    which drags tip and ref together, never reaches the cursor. Finger
    articulation during an arm move is discarded by design, not queued.

    Representation: virtual = tip - offset. ARM-STILL keeps the offset
    constant; ARM-MOVING folds every tip delta into the offset, so the
    virtual holds exactly and release resumes with no jump. Only the
    virtual's DELTAS matter downstream (HandPointerPipeline is
    anchor-relative), so the offset coexists with the anchor, the
    stillness gate, and the recenter gesture.

    Detection is positional over a short window (never instantaneous
    velocity - see the stillness-gate rationale above), with the same
    trimmed-mean noise adaptation. Arm-onset inevitably leaks 2-3 frames
    of motion; the transition rolls the window's tip motion back into the
    offset, and the downstream gate - which holds pending excursions
    unmoved and absorbs receding ones with zero residue - turns a
    from-rest arm move into literally zero cursor motion.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.moving = False
        self.offset = np.zeros(2)
        self._last_tip: np.ndarray | None = None
        self._gap_tip: np.ndarray | None = None  # last tip before a loss
        self._moving_since: float | None = None
        self._ref_hist: deque[np.ndarray] = deque(maxlen=3)  # spike filter
        self._win: deque[tuple[float, np.ndarray]] = deque()  # (t, filtered ref)
        self._tip_win: deque[tuple[float, np.ndarray]] = deque()  # (t, raw tip)
        # Noise scale from still-period ref deltas only: during an arm move
        # every delta IS movement, so the trimmed mean can't save a
        # contaminated deque the way it does downstream.
        self._deltas: deque[float] = deque(maxlen=60)
        self._prev_gref: np.ndarray | None = None
        self._enter_pending = 0
        self._enter_eff = cfg.hand_pure_arm_enter    # effective thresholds (HUD)
        self._settle_eff = cfg.hand_pure_arm_settle
        self._disp = 0.0  # last window displacement (HUD)
        self._reacquire = False
        # When the gate last released to STILL: rollback never reaches
        # behind this, so motion a previous MOVING period already absorbed
        # into the offset can't be subtracted twice (double-subtraction
        # flings the virtual backwards).
        self._still_t = float("-inf")

    @staticmethod
    def _at_window_start(win, now, span, floor=float("-inf")):
        """First sample at or after max(now-span, floor) (newest sample
        before it if the window is still filling)."""
        target = max(now - span, floor)
        p0 = None
        for t, p in win:
            p0 = p
            if t >= target:
                break
        return p0

    @staticmethod
    def _win_mean(win, t0, t1):
        """Mean of samples in [t0, t1] (None if the span is empty)."""
        pts = [p for t, p in win if t0 <= t <= t1]
        return np.mean(np.stack(pts), axis=0) if pts else None

    def update(self, tip: np.ndarray, ref: np.ndarray, now: float) -> np.ndarray:
        cfg = self.cfg
        if self._reacquire:
            # Hand re-entered the frame: wherever it reappeared, that
            # relocation IS arm motion - absorb the whole gap so the
            # cursor holds, and start MOVING until settle confirms
            # (reacquisition frames are re-detection-noisy).
            if self._gap_tip is not None:
                self.offset += tip - self._gap_tip
            self._gap_tip = None
            self._reacquire = False
            self.moving = True
            self._moving_since = now

        self._ref_hist.append(ref)
        gref = np.median(np.stack(self._ref_hist), axis=0)
        keep = max(cfg.hand_pure_arm_window_s, cfg.hand_pure_arm_settle_s,
                   cfg.hand_pure_arm_slow_window_s)
        self._win.append((now, gref))
        while self._win and now - self._win[0][0] > keep:
            self._win.popleft()
        self._tip_win.append((now, tip.copy()))
        while self._tip_win and now - self._tip_win[0][0] > cfg.hand_pure_rollback_s:
            self._tip_win.popleft()
        if self._prev_gref is not None and not self.moving:
            self._deltas.append(float(np.linalg.norm(gref - self._prev_gref)))
        self._prev_gref = gref
        if len(self._deltas) >= 20:
            d = np.fromiter(self._deltas, float)
            lo, hi = np.percentile(d, [25, 75])
            mid = d[(d >= lo) & (d <= hi)]
            nd = float(mid.mean()) if len(mid) else 0.0
            self._enter_eff = max(cfg.hand_pure_arm_enter,
                                  cfg.hand_pure_arm_noise_enter * nd)
            self._settle_eff = max(cfg.hand_pure_arm_settle,
                                   cfg.hand_pure_arm_noise_settle * nd)

        w = cfg.hand_pure_arm_window_s
        ref0 = self._at_window_start(self._win, now, w)
        tip0 = self._at_window_start(self._tip_win, now, w)
        self._disp = (float(np.linalg.norm(gref - ref0))
                      if ref0 is not None else 0.0)

        if self.moving:
            if self._last_tip is not None:
                self.offset += tip - self._last_tip  # virtual frozen exactly
            settled = False
            if (self._moving_since is not None
                    and now - self._moving_since >= cfg.hand_pure_arm_min_move_s
                    and self._win
                    and now - self._win[0][0] >= 0.8 * cfg.hand_pure_arm_settle_s):
                recent = np.stack([p for t, p in self._win
                                   if t >= now - cfg.hand_pure_arm_settle_s])
                spread = float(np.max(
                    np.linalg.norm(recent - recent.mean(axis=0), axis=1)))
                # Spread-based release: noise can't fake a settle mid-move
                # the way a single small frame delta could.
                settled = spread < self._settle_eff
            if settled:
                self.moving = False
                self._moving_since = None
                self._enter_pending = 0
                self._still_t = now
        else:
            tip_disp = (float(np.linalg.norm(tip - tip0))
                        if tip0 is not None else 0.0)
            # Coherence: arm translation moves ref and tip together
            # (ratio ~1), a finger wag only rocks the palm (~0.1-0.3) -
            # without this test, vigorous deliberate pointing would
            # intermittently freeze the cursor. When the tip is barely
            # moving the test passes trivially and ref motion decides.
            fast = (self._disp > self._enter_eff
                    and self._disp > cfg.hand_pure_arm_coherence * tip_disp)
            # Slow-motion test: a slow arm glide's per-window displacement
            # hides under the noise-adapted threshold, but jitter's net
            # displacement does NOT grow with the window while a real
            # glide's does - so the SAME threshold over a longer baseline
            # separates them (the downstream gate's horizon guard,
            # inverted). Compared via short sub-window MEANS at each end,
            # not endpoint samples: the noise-delta-calibrated threshold
            # is far too tight for two raw samples 0.6s apart (their
            # difference has full marginal variance), and every false
            # onset injects a permanent rollback step into the virtual.
            sw = cfg.hand_pure_arm_slow_window_s
            slow = False
            if self._win and now - self._win[0][0] >= 0.8 * sw:
                m = cfg.hand_pure_arm_window_s
                rs = self._win_mean(self._win, now - sw, now - sw + m)
                re = self._win_mean(self._win, now - m, now)
                ta = self._win_mean(self._tip_win, now - sw, now - sw + m)
                tb = self._win_mean(self._tip_win, now - m, now)
                if rs is not None and re is not None:
                    sdisp = float(np.linalg.norm(re - rs))
                    stip_disp = (float(np.linalg.norm(tb - ta))
                                 if ta is not None and tb is not None else 0.0)
                    slow = (sdisp > self._enter_eff
                            and sdisp > cfg.hand_pure_arm_coherence * stip_disp)
            # No onsets until the noise estimator has samples: on a noisy
            # setup the floors are too tight for the first ~0.7s, and
            # every false onset injects a permanent rollback step into
            # the virtual (startup cursor wander).
            if (fast or slow) and len(self._deltas) >= 20:
                self._enter_pending += 1
            else:
                self._enter_pending = 0
            if self._enter_pending >= cfg.hand_pure_arm_enter_frames:
                # Onset: rewind the leaked frames over the window that saw
                # the motion, but never past the last release (_still_t):
                # motion absorbed by a previous MOVING period must not be
                # subtracted twice. Slow-only detection means the whole
                # span moved coherently, so rewinding it rewinds arm
                # motion, not finger intent - and its start point is a
                # sub-window mean, so a noise spike can't become the
                # rollback target. The receding excursion is absorbed
                # downstream with zero residue when the gate is locked;
                # when unlocked the One Euro turns it into a small wiggle
                # - do NOT reset the smoother here, that jump would be
                # worse.
                w = cfg.hand_pure_arm_window_s
                span = w if fast else sw
                if self._still_t > now - span:
                    # Re-firing shortly after a release: the exact
                    # continuity point is the tip at release time. A mean
                    # CENTERED on it stays unbiased under a steady glide
                    # (the absorbed pre-half cancels the leaked post-half)
                    # instead of handing back half a sub-window of leak
                    # per settle/re-freeze cycle - that bias ratchets.
                    back = self._win_mean(self._tip_win,
                                          self._still_t - w / 2,
                                          self._still_t + w / 2)
                elif fast:
                    back = self._at_window_start(self._tip_win, now, w)
                else:
                    back = self._win_mean(self._tip_win, now - sw,
                                          now - sw + w)
                if back is not None:
                    self.offset += tip - back
                self.moving = True
                self._moving_since = now
                self._enter_pending = 0

        self._last_tip = tip.copy()
        return tip - self.offset

    def lost(self):
        """Right hand left the frame: hold the virtual, remember the last
        tip so the reacquisition gap can be absorbed (keep the offset)."""
        if self._last_tip is not None:
            self._gap_tip = self._last_tip
        self._last_tip = None
        self.moving = False
        self._moving_since = None
        self._enter_pending = 0
        self._ref_hist.clear()  # median across the gap would mix positions
        self._win.clear()
        self._tip_win.clear()
        self._prev_gref = None  # a delta across the gap isn't a noise sample
        self._reacquire = True

    def reset(self):
        """Pause toggled: same handling as a lost pointer."""
        self.lost()

    def info(self) -> str:
        """[ARM] HUD line for threshold tuning."""
        if self.moving:
            return (f"arm  MOVING (frozen)  disp {self._disp:.4f} "
                    f"(spread < {self._settle_eff:.4f} releases)")
        return (f"arm  still            disp {self._disp:.4f} "
                f"(> {self._enter_eff:.4f} freezes)")
