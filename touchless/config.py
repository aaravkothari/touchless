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

    # --- Calibration ---
    calib_grid: int = 4            # normal-posture stage: 4x4 = 16 dots
    calib_stage_grid: int = 3      # lean-back / lean-in stages: 3x3 = 9 dots
    calib_margin: float = 0.08     # fraction of screen kept as border
    calib_glide_s: float = 0.35    # dot glides to its next position (eyes follow)
    calib_capture_s: float = 1.2   # time spent collecting samples per dot
    calib_timeout_s: float = 5.0   # max wait for gaze to stabilize on a dot
    calib_stability_std: float = 0.02  # gaze std (rolling 0.4s) below this = "landed"
    calib_file: str = "calibration.json"
    calib_mad_z: float = 2.5       # per-dot outlier rejection threshold
    # Ridge lambda is picked by leave-one-out CV over this grid:
    ridge_lambdas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)

    # --- Prediction robustness ---
    pred_clamp: float = 0.15       # raw predictions clamped to [-x, 1+x]
    blink_gate: float = 0.35       # blink score above this: hold cursor, drop
                                   # calibration samples (blinks corrupt gaze)

    # --- Cursor ---
    screen_inset_px: int = 10      # keep cursor away from corners so
                                   # pyautogui's fail-safe stays user-triggered

    # --- Dwell click ---
    dwell_radius_px: int = 45      # cursor must stay inside this circle
    dwell_time_s: float = 1.0      # ...for this long to click
    dwell_cooldown_s: float = 1.5  # refractory period after a click

    # --- Blink click (uses the model's eyeBlink blendshape, 0..1) ---
    blink_closed_threshold: float = 0.5
    blink_min_s: float = 0.25      # deliberate blink, not a reflex blink
    blink_max_s: float = 1.5       # longer than this = probably resting eyes
    blink_cooldown_s: float = 1.0
