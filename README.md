# touchless

Control your mouse cursor with your eyes + head using a plain webcam.

This is the Python MVP of a project that will eventually become a Tauri app.
The goal of this stage is to **finalize the tracking/control pipeline** —
calibration math, smoothing, click gestures — before any app-shell work.

---

## Quick start

```powershell
# one-time setup
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. sanity-check your camera and lighting (no cursor control)
.\.venv\Scripts\python.exe -m touchless preview

# 2. calibrate (fullscreen dot sequence + accuracy check, ~1 min)
.\.venv\Scripts\python.exe -m touchless calibrate

# 3. drive the cursor
.\.venv\Scripts\python.exe -m touchless run
```

On first run a ~3.7 MB MediaPipe model is downloaded automatically into `models/`.

### Safety / emergency stop

- **Slam the mouse into the top-left corner** → the app stops instantly
  (pyautogui's fail-safe). This works even when no window is focused.
- `q` quits and `space` pauses, but only while the small preview window is focused.
- Clicking is **off by default** — the app only moves the cursor until you
  opt into a click mode.

---

## Commands

| Command | What it does |
|---|---|
| `python -m touchless preview` | Camera + tracking overlay + live numbers. If calibrated, also shows the predicted cursor point in a mini screen-rect. No cursor control. |
| `python -m touchless calibrate` | Fullscreen 16-dot calibration, then a 5-dot accuracy check. You see your real error in px before deciding to save (`Enter`), redo (`r`), or abort (`Esc`). |
| `python -m touchless run [--click off\|dwell\|blink] [--log FILE.csv]` | Drive the cursor with the saved calibration. `--log` records features + predictions for offline analysis. |
| `--camera N` (any command) | Use webcam index N if you have more than one. |

### Click methods

- **`off`** (default) — cursor movement only.
- **`dwell`** — hold the cursor still (within 45 px) for 1 s → left click.
- **`blink`** — deliberately close both eyes for 0.25–1.5 s → left click.
  Normal reflex blinks (~0.1 s) are ignored. Uses the model's `eyeBlink`
  blendshape score, visible live in `preview`.

---

## How it works

```
webcam frame, 1280x720 (OpenCV)
   │
   ▼
MediaPipe FaceLandmarker
   • 478 landmarks (drawn in preview)
   • 52 blendshapes  ──►  eyeLook* (trained gaze coefficients), eyeBlink*
   • facial transformation matrix  ──►  head yaw/pitch + translation
   │
   ▼
feature vector (tracking.py)
   [gaze_x, gaze_y, yaw, pitch, tx, ty]   — eyes AND head, fused
   │
   ▼
calibration map (calibration.py)
   z-score standardization → [1, z, z²] expansion (13 terms)
   → ridge regression, lambda chosen by leave-one-out CV
   → normalized screen (x, y)
   │
   ▼
robustness (app.py): clamp → median-of-3 → One Euro filter (smoothing.py)
   │
   ▼
cursor (mouse.py) + click gestures (clicker.py)
```

### Why each piece exists

**Model-learned features, not geometry.** Gaze comes from MediaPipe's
`eyeLook*` blendshapes (a network trained on humans looking around) and head
pose from its facial transformation matrix. Earlier versions hand-computed
iris offsets and solvePnP head pose — both were noisy enough to sink the
whole pipeline. Head translation (tx, ty) is included so the mapping
tolerates you shifting in your chair.

**Standardization before regression.** The features mix scales (blendshapes
~0.3, yaw ~30°, translation ~cm). Ridge regression with one penalty across
raw scales silently crushes the small-scale signals — z-scoring first puts
every signal on equal footing. The mean/std are stored in `calibration.json`.

**A deliberately small model.** The expansion is linear + squared terms only
(13 terms from 6 features) fit on 16 points. A full quadratic expansion has
more terms than calibration dots and extrapolates violently between them —
that bug looked like "the cursor flies off in some screen regions".

**Cross-validated regularization.** The ridge lambda is picked by
leave-one-out CV on your calibration data instead of a hardcoded guess.

**Validation before saving.** After fitting, you look at 5 fresh dots the
fit has never seen, with the predicted point drawn live. You see mean/worst
error in pixels and choose to save or redo. Garbage calibrations no longer
save silently.

**Robust prediction path.** Raw predictions are clamped (an off-screen
fling can't wind up the filter), median-of-3 filtered (kills single-frame
spikes), then One Euro smoothed (steady when you hold still, responsive
when you flick — the same filter class VR controllers use).

---

## Project layout

```
touchless/
├── touchless/
│   ├── __main__.py     CLI (argparse) — start reading here
│   ├── app.py          main loops for preview / calibrate / run
│   ├── config.py       ALL tunable parameters, documented
│   ├── tracking.py     webcam + MediaPipe → FaceSample (the core contract)
│   ├── calibration.py  16-dot routine, ridge fit + LOOCV, validation pass
│   ├── smoothing.py    One Euro filter
│   ├── mouse.py        OS cursor wrapper (pyautogui)
│   └── clicker.py      dwell + blink click state machines
├── models/             auto-downloaded MediaPipe model (gitignored)
├── calibration.json    your personal calibration (gitignored, versioned)
└── requirements.txt
```

The key interface is `FaceSample` in `tracking.py`. Everything downstream
only knows about that dataclass — so when this becomes a Tauri app, you can
swap the perception layer (or move it behind a socket) without touching the
calibration/smoothing/click logic.

---

## Testing & iterating

### 1. Verify tracking (`preview`)

- `gaze x/y` should move when you look around **without moving your head**.
- `yaw/pitch` should track head turns with believable numbers (roughly ±30°).
- `blink` should jump toward 1.0 when you close your eyes.
- fps should be ≥ 20 at 720p. If not, set `frame_width/height` to 640×480
  in `config.py`.

If tracking is jittery here, fix it *here* first (lighting, camera position)
— no amount of downstream tuning saves bad input. Face the light source;
avoid strong backlight.

### 2. Calibrate and read the validation numbers

The check screen reports mean/worst error in pixels. Rough guide on a
1920-wide screen: **< 100 px mean is good** for webcam tracking, ~150 px is
usable with dwell clicking, worse → hit `r` and redo (check posture and
lighting first). Recalibrate whenever you change posture, chair height, or
lighting significantly — it's under a minute.

### 3. End-to-end drill

Open Paint fullscreen, `run --click off`, try to park the cursor on each
corner and the center. Tight cluster = good. Then turn on dwell and try
actually using the machine.

### 4. Offline iteration (no face required)

```powershell
.\.venv\Scripts\python.exe -m touchless run --log session.csv
```

records `[t, features..., raw_x/y, smooth_x/y]` per frame. Replay it in a
notebook to evaluate smoothing/mapping changes against recorded sessions
instead of your live face.

### Tuning loop

All knobs live in `touchless/config.py` with comments. The ones you'll
actually touch:

| Symptom | Knob | Direction |
|---|---|---|
| Cursor jitters when holding still | `smooth_min_cutoff` | lower (e.g. 0.4) |
| Cursor lags behind fast glances | `smooth_beta` | higher (e.g. 1.0) |
| Dwell clicks fire accidentally | `dwell_time_s` / `dwell_radius_px` | raise / lower |
| Blinks not detected | `blink_closed_threshold` | lower (watch `blink` in preview) |
| Validation error high everywhere | recalibrate; check lighting | — |
| Fit feels over/under-constrained | `ridge_lambdas` grid | widen |

### Known limitations (MVP)

- Physics still caps webcam gaze at region-level precision; the head-pose
  component is what gives you fine control. The planned next lever is
  **pursuit calibration** (follow a moving dot → hundreds of training
  samples instead of 16) if the current accuracy isn't enough.
- Single monitor assumed (uses primary screen size).
- Windows-focused (tested there); macOS needs accessibility permissions for
  pyautogui, Linux needs X11.

---

## Roadmap → Tauri

1. **Finalize pipeline here** — accuracy, click UX, smoothing constants.
   This is the hard, iterate-heavy part. *(you are here)*
2. **Split engine from UI** — run the tracker as a headless process emitting
   `(x, y, click)` over a local WebSocket; a first "remote control" client
   proves the interface.
3. **Tauri app** — frontend = settings/calibration UI; engine = either
   - the Python tracker compiled with PyInstaller, bundled as a
     [Tauri sidecar](https://tauri.app/develop/sidecar/) (fast path), or
   - a Rust rewrite using ONNX Runtime + `enigo` for cursor control
     (clean path, more work).

Because of step 2's interface, both engine options slot in without changing
the app.
