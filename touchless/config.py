"""Central tunables. Everything you'd want to tweak while iterating lives here."""

from dataclasses import dataclass


@dataclass
class Config:
    # --- Camera ---
    camera_index: int = 0
    frame_width: int = 1280    # 720p: more pixels on the iris = better signal.
    frame_height: int = 720    # Drop to 640x480 if preview fps < 20.

    # --- Smoothing (One Euro filter) ---
    # Lower min_cutoff = smoother but laggier at rest.
    # Higher beta = less lag during fast movement (at the cost of jitter).
    smooth_min_cutoff: float = 0.5
    smooth_beta: float = 0.6

    # --- Pursuit calibration (follow the moving cursor) ---
    pursuit_phase_s: float = 60.0        # seconds per phase (2 phases)
    pursuit_speed: tuple[float, float] = (0.12, 0.30)  # dot speed, screen/s
    pursuit_hold_prob: float = 0.5       # chance of pausing at a waypoint
    pursuit_hold_s: tuple[float, float] = (0.2, 0.6)
    pursuit_edge_every: int = 4          # every Nth waypoint hits an edge/corner
    calib_margin: float = 0.04           # dot keeps this off the screen border

    # --- Model fitting ---
    model_file: str = "model.pkl"
    data_file: str = "gaze_data.npz"     # raw pursuit session (for `retrain`)
    lag_grid_ms: tuple[int, ...] = (0, 50, 100, 150, 200, 250)
    holdout_frac: float = 0.2            # temporal tail of each phase held out
    blink_after_s: float = 0.2           # also drop this long after each blink
    ridge_lambdas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)

    # --- Validation dots (after fitting) ---
    calib_glide_s: float = 0.35    # dot glides to its next position (eyes follow)
    calib_capture_s: float = 1.2   # time spent collecting samples per dot
    calib_timeout_s: float = 5.0   # max wait for gaze to stabilize on a dot
    calib_stability_std: float = 0.02  # gaze std (rolling 0.4s) below this = "landed"
    calib_mad_z: float = 2.5       # per-dot outlier rejection threshold

    # --- Prediction robustness ---
    pred_clamp: float = 0.15       # raw predictions clamped to [-x, 1+x]
    blink_gate: float = 0.35       # blink score above this: hold cursor, drop
                                   # calibration samples (blinks corrupt gaze)

    # --- Cursor ---
    screen_inset_px: int = 10      # keep cursor away from corners so
                                   # pyautogui's fail-safe stays user-triggered

    # --- Hand mode (right hand moves; left hand + face are the controls) ---
    hand_gain_x: float = 6.0       # horizontal: screen-widths per camera-width of travel
    hand_gain_y: float = 4.5       # vertical (screens are wide - x needs more reach)
    # Hand mode gets its own One Euro tuning: high gain amplifies hand
    # tremor, so smoothing at rest is much heavier than in gaze mode while
    # beta keeps fast sweeps responsive.
    hand_smooth_min_cutoff: float = 0.25
    hand_smooth_beta: float = 1.5  # high = fast sweeps cut through the filter
    # Velocity-estimate cutoff for the beta term. Below the 1 Hz default:
    # per-frame landmark noise has huge instantaneous velocity, and a fast
    # d_cutoff lets that noise fire the beta boost - reducing smoothing
    # exactly when the hand is still (the unlocked-state wobble).
    hand_smooth_d_cutoff: float = 0.5
    # Hand mode runs its own lower camera resolution: fingertips don't need
    # 720p (that was for iris pixels) and the smaller frames roughly double
    # the frame rate of the dual-model stack.
    hand_frame_width: int = 640
    hand_frame_height: int = 480
    hand_labels_flipped: bool = False  # flip if preview labels your hands wrong
    # --- Hand-wrist mode (run --input hand-wrist) ---
    # Pointer = right index tip RELATIVE to the wrist, so moving the whole
    # hand around does nothing - only articulating the finger moves the
    # cursor. The rel vector is in hand-size units (not camera widths), so
    # these gains live on a different scale than hand_gain_*:
    # screens per hand-size of fingertip travel.
    hand_wrist_gain_x: float = 2.5     # side-to-side finger wag has small range
    hand_wrist_gain_y: float = 1.6     # curl/extend has more range than the wag
    # Reference point: the wrist landmark sits at the base of the palm,
    # which reads as "too high" - the reference is extrapolated this many
    # hand-sizes below it, onto the forearm.
    hand_wrist_ref_drop: float = 0.6
    # The extrapolated ref is 1.6*wrist - 0.6*MCP: ~3x the noise variance
    # of a single landmark, multiplied straight into pointer_rel. EMA it
    # (new-sample weight below). The ref only moves when the whole hand
    # translates - which wrist mode deliberately ignores - so the lag this
    # adds is nearly free; the noise cut (~30% of pointer_rel std) is not.
    hand_wrist_ref_ema: float = 0.5
    # The rel signal is noisier than the absolute pointer (two jittering
    # landmarks + normalization), so rest smoothing is much heavier here.
    hand_wrist_smooth_min_cutoff: float = 0.12
    hand_wrist_smooth_beta: float = 1.2
    hand_wrist_smooth_d_cutoff: float = 0.5  # see hand_smooth_d_cutoff
    # Cursor moves smaller than this many pixels are swallowed (hysteresis:
    # slow drifts still accumulate and get through). Kills at-rest shake.
    hand_deadzone_px: int = 5
    # Stillness gate: landmark drift ("the dots recalibrating") is unbounded
    # over time, so no deadzone can absorb it. When the finger is judged
    # still, the anchor tracks the pointer 1:1 - drift is eaten by the
    # anchor, cursor freezes. Detection is POSITIONAL, not velocity-based:
    # per-frame jitter makes instantaneous speed look huge (especially in
    # wrist mode's hand-size units) but positions still cluster tightly.
    # All distances are gain-scaled screen fractions.
    hand_still_enter: float = 0.035   # window spread below this = lock
    # The exit threshold doubles as the minimum deliberate move while
    # locked, so keep the floor tight - robust detection (median gate
    # signal + sustained-frames exit + noise adaptation) is what makes a
    # tight floor safe, not headroom.
    hand_still_exit: float = 0.035    # stray this far from the lock = unlock
    # Slow-creep escape: a deliberate move slower than the lock EMA would
    # otherwise be absorbed forever ("cursor stuck"). Sitting beyond half
    # the exit threshold for this long unlocks too.
    hand_still_creep_s: float = 0.7
    # Relock evidence horizon: one spread-window (0.35s) cannot tell a
    # slow deliberate move from stillness - the per-window displacement is
    # below the noise floor. Locking additionally requires the net
    # displacement across this longer horizon to be near zero, so a
    # sustained slow move keeps the gate open and keeps flowing.
    hand_still_horizon_s: float = 1.5
    hand_still_net_mult: float = 0.5  # net displacement must be below
                                      # this fraction of the enter threshold
    hand_still_window_s: float = 0.35  # how much recent history the spread uses
    hand_still_lock_adapt: float = 0.02  # lock point EMA/frame: slow drift
                                         # tracks, deliberate moves don't
    hand_still_exit_frames: int = 3   # excursion must persist this many
                                      # consecutive frames to unlock (a
                                      # 1-2 frame noise burst can't)
    # Re-detection step absorption: when MediaPipe re-runs palm detection
    # the landmarks re-solve to a slightly different answer - a PERSISTENT
    # step, which would otherwise read as a real move and jump the cursor.
    # A sustained excursion that is RESTING at its new position (tight
    # cluster, no ongoing motion) and smaller than step_mult * exit is
    # treated as the dots recalculating and absorbed. Real moves keep
    # moving through the check and unlock as usual.
    hand_still_step_mult: float = 3.0
    # Noise-adaptive thresholds: the pipeline estimates the setup's noise
    # scale online (25th percentile of recent frame-to-frame gate deltas -
    # deliberate movement can't inflate a low quantile) and uses
    # max(floor above, factor * noise) so a noisy webcam/lighting setup
    # widens the gate instead of leaving it flapping. Clean setups stay on
    # the floors.
    # Multiples of the noise scale (trimmed-mean frame delta of the gate
    # signal, ~0.82 of its per-axis sigma). exit lands at ~4.5 sigma: a
    # sustained 3-frame crossing is noise-impossible, only a real move.
    hand_still_noise_enter: float = 4.0
    hand_still_noise_exit: float = 5.5
    pinch_click_threshold: float = 0.35   # pinch dist below this = button DOWN
    pinch_release_threshold: float = 0.45  # ...and back above this = button UP
    pinch_refractory_s: float = 0.15  # min gap between a release and next press
    tongue_jaw_gate: float = 0.25  # jawOpen needed before tongue is even checked
    tongue_threshold: float = 0.45  # tongue-pixel fraction above this = tongue out
    tongue_hold_s: float = 0.2     # tongue must hold this long to recenter
    tongue_cooldown_s: float = 1.0
    face_every_n: int = 3          # run the face model every Nth frame in hand mode
                                   # (tongue detection doesn't need full rate)

    # --- Dwell click ---
    dwell_radius_px: int = 45      # cursor must stay inside this circle
    dwell_time_s: float = 1.0      # ...for this long to click
    dwell_cooldown_s: float = 1.5  # refractory period after a click

    # --- Blink click (uses the model's eyeBlink blendshape, 0..1) ---
    blink_closed_threshold: float = 0.5
    blink_min_s: float = 0.25      # deliberate blink, not a reflex blink
    blink_max_s: float = 1.5       # longer than this = probably resting eyes
    blink_cooldown_s: float = 1.0
