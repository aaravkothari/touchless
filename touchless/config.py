"""Central tunables. Everything you'd want to tweak while iterating lives here."""

from dataclasses import dataclass


@dataclass
class Config:
    # --- Camera ---
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # --- Smoothing (One Euro filter) ---
    # Lower min_cutoff = smoother but laggier at rest.
    # Higher beta = less lag during fast movement (at the cost of jitter).
    smooth_min_cutoff: float = 0.8
    smooth_beta: float = 0.6

    # --- Calibration ---
    calib_grid: int = 3            # 3 -> 3x3 = 9 points
    calib_margin: float = 0.08     # fraction of screen kept as border
    calib_settle_s: float = 1.0    # time to let your gaze land on the dot
    calib_capture_s: float = 1.2   # time spent collecting samples per dot
    calib_file: str = "calibration.json"
    ridge_lambda: float = 1e-3     # regularization for the polynomial fit

    # --- Cursor ---
    screen_inset_px: int = 10      # keep cursor away from corners so
                                   # pyautogui's fail-safe stays user-triggered

    # --- Dwell click ---
    dwell_radius_px: int = 45      # cursor must stay inside this circle
    dwell_time_s: float = 1.0      # ...for this long to click
    dwell_cooldown_s: float = 1.5  # refractory period after a click

    # --- Blink click ---
    ear_closed_threshold: float = 0.18  # eye aspect ratio below = closed
    blink_min_s: float = 0.25      # deliberate blink, not a reflex blink
    blink_max_s: float = 1.5       # longer than this = probably resting eyes
    blink_cooldown_s: float = 1.0
