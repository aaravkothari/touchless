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

# 2. calibrate (3-posture dot sequence + accuracy check, ~2 min)
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
| `python -m touchless preview` | Camera + full telemetry (gaze, head angles, position, depth bar). If calibrated, also shows the predicted cursor point in a mini screen-rect. No cursor control. |
| `python -m touchless calibrate` | 3-posture calibration (16 dots sitting normally, 9 leaned back, 9 leaned in), then an accuracy check at normal posture *and* leaned back. You see your real error in px before deciding to save (`Enter`), redo (`r`), or abort (`Esc`). |
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
   • facial transformation matrix  ──►  yaw/pitch/roll + x/y/depth position
   │
   ▼
feature vector (tracking.py)
   [gaze_x, gaze_y, yaw, pitch, roll, tx, ty, tz]   — eyes AND full head pose
   │
   ▼
calibration map (calibration.py)
   17 design terms incl. tz×(gaze,yaw,pitch) depth interactions
   → columns z-scored → ridge regression, lambda by leave-one-out CV
   → normalized screen (x, y)
   │
   ▼
robustness (app.py): blink gate → clamp → median-of-3 → One Euro filter
   │
   ▼
cursor (mouse.py) + click gestures (clicker.py)
```

### Why each piece exists

**Model-learned features, not geometry.** Gaze comes from MediaPipe's
`eyeLook*` blendshapes (a network trained on humans looking around) and full
head pose — yaw/pitch/roll plus x/y/depth position — from its facial
transformation matrix. Earlier versions hand-computed iris offsets and
solvePnP head pose; both were noisy enough to sink the whole pipeline.

**Depth interactions make it posture-proof.** Physics: how far your gaze
lands on screen scales with how far you sit from it (displacement ≈
distance × tan(angle)). The model therefore has explicit `tz × gaze` and
`tz × yaw/pitch` terms — and calibration runs at **three postures** (normal,
leaned back, leaned in) so those terms are actually learnable. A
single-posture calibration has zero variance in head position; no model
form can learn position invariance from it.

**Gaze-arrival gating in calibration.** Capture at each dot starts only
once your gaze has *landed* — a rolling 0.4 s window of gaze features must
go quiet (and eyes must be open) before samples count, instead of a fixed
timer that records your eyes mid-flight. The dot also glides between
positions so your eyes naturally track it. The dot turns green when capture
actually begins.

**Standardization + cross-validated ridge.** Design-matrix columns are
z-scored (mixed scales otherwise crush the small signals; stats stored in
`calibration.json`), the ridge lambda is picked by leave-one-out CV on your
data, and columns are clipped at predict time so the interaction terms
can't explode outside the calibrated range.

**Validation before saving.** After fitting, you look at 5 fresh dots at
normal posture plus one leaned back, with the predicted point drawn live.
You see mean/worst error in pixels — including the posture-change error —
and choose to save or redo. Garbage calibrations no longer save silently.

**Robust prediction path.** While you blink the cursor holds still (blinks
send the gaze blendshapes haywire — this was the biggest jitter source).
Raw predictions are clamped (an off-screen fling can't wind up the filter),
median-of-3 filtered (kills single-frame spikes), then One Euro smoothed
(steady when you hold still, responsive when you flick — the same filter
class VR controllers use).

---

## Project layout

```
touchless/
├── touchless/
│   ├── __main__.py     CLI (argparse) — start reading here
│   ├── app.py          main loops for preview / calibrate / run
│   ├── config.py       ALL tunable parameters, documented
│   ├── tracking.py     webcam + MediaPipe → FaceSample (the core contract)
│   ├── calibration.py  3-posture routine, depth-aware fit, validation pass
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

The telemetry panel shows everything being tracked, live:

```
preview - q to quit   29.1 fps
gaze  x +0.12  y -0.05   blink 0.08 open
head  yaw +3.2  pitch -8.1  roll +1.4      <- turn/nod/tilt your head
pos   x +2.1cm  y -0.5cm  depth 61.0cm     <- slide/lean around
[############      ] depth                 <- bar moves as you lean
```

- `gaze x/y` should move when you look around **without moving your head**.
- `yaw/pitch/roll` should track head turns/nods/tilts with believable numbers.
- `depth` should grow when you lean back, shrink when you lean in.
- `blink` should jump toward 1.0 when you close your eyes.
- fps should be ≥ 20 at 720p. If not, set `frame_width/height` to 640×480
  in `config.py`.

If tracking is jittery here, fix it *here* first (lighting, camera position)
— no amount of downstream tuning saves bad input. Face the light source;
avoid strong backlight.

### 2. Calibrate and read the validation numbers

The check screen reports mean/worst error at normal posture **and** the
leaned-back error separately. Rough guide on a 1920-wide screen: **< 100 px
mean is good** for webcam tracking, ~150 px is usable with dwell clicking,
worse → hit `r` and redo (check lighting first). During the lean-back and
lean-in stages, genuinely change your posture and hold it — that variance
is what makes the mapping survive you moving later. Thanks to the 3-posture
protocol you should NOT need to recalibrate just because you shifted; redo
it when lighting changes drastically or accuracy visibly degrades.

### 3. End-to-end drill

Open Paint fullscreen, `run --click off`, try to park the cursor on each
corner and the center. Tight cluster = good. Then turn on dwell and try
actually using the machine.

### 4. Offline iteration (no face required)

```powershell
.\.venv\Scripts\python.exe -m touchless run --log session.csv
```

records `[t, features..., blink, raw_x/y, smooth_x/y]` per frame. Replay it
in a notebook to evaluate smoothing/mapping changes against recorded
sessions instead of your live face.

### Tuning loop

All knobs live in `touchless/config.py` with comments. The ones you'll
actually touch:

| Symptom | Knob | Direction |
|---|---|---|
| Cursor jitters when holding still | `smooth_min_cutoff` | lower (e.g. 0.3) |
| Cursor lags behind fast glances | `smooth_beta` | higher (e.g. 1.0) |
| Cursor freezes too often (squinting reads as blink) | `blink_gate` | raise (watch `blink` in preview) |
| Calibration dots take forever to turn green | `calib_stability_std` | raise (e.g. 0.03) |
| Dots turn green before your eyes arrive | `calib_stability_std` | lower |
| Dwell clicks fire accidentally | `dwell_time_s` / `dwell_radius_px` | raise / lower |
| Blink-clicks not detected | `blink_closed_threshold` | lower |
| Validation error high everywhere | recalibrate; check lighting | — |
| Accuracy dies when leaning | redo calibration with *bigger* posture differences between stages | — |

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
