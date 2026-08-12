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

    # --- Hand mode (right index finger drives the cursor) ---
    hand_gain: float = 1.8         # screen-widths moved per camera-width of finger travel
    hand_labels_flipped: bool = True   # MediaPipe assumes a mirrored image; our
                                       # frames aren't, so labels are swapped
    fist_hold_s: float = 0.15      # left fist must hold this long to recenter
    fist_cooldown_s: float = 0.6
    pinch_click_threshold: float = 0.35  # thumb-index dist / hand size below = pinch
    pinch_cooldown_s: float = 0.5

    # --- Dwell click ---
    dwell_radius_px: int = 45      # cursor must stay inside this circle
    dwell_time_s: float = 1.0      # ...for this long to click
    dwell_cooldown_s: float = 1.5  # refractory period after a click

    # --- Blink click (uses the model's eyeBlink blendshape, 0..1) ---
    blink_closed_threshold: float = 0.5
    blink_min_s: float = 0.25      # deliberate blink, not a reflex blink
    blink_max_s: float = 1.5       # longer than this = probably resting eyes
    blink_cooldown_s: float = 1.0
