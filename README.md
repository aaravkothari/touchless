# touchless

Control your mouse cursor with your eyes (or head) using a plain webcam.

This is the Python MVP of a project that will eventually become a Tauri app.
The goal of this stage is to **finalize the tracking/control pipeline** —
calibration math, smoothing, click gestures — before any app-shell work.

---

## Quick start

```powershell
# one-time setup
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. sanity-check your camera and lighting (no mouse control)
.\.venv\Scripts\python.exe -m touchless preview

# 2. calibrate (fullscreen dot sequence, ~25 seconds)
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
| `python -m touchless preview` | Show the camera with tracking overlay + live numbers. No cursor control. |
| `python -m touchless calibrate [--mode gaze\|head]` | Fullscreen 9-dot calibration. Writes `calibration.json`. |
| `python -m touchless run [--click off\|dwell\|blink]` | Drive the cursor using the saved calibration. |
| `--camera N` (any command) | Use webcam index N if you have more than one. |

### Modes

- **`gaze`** (default) — iris position + head pose. More "eyes move the mouse",
  but webcam gaze is inherently coarse: expect region-level accuracy, not
  pixel-level.
- **`head`** — head pose only (point your nose at the target). Noticeably more
  stable; many hands-free-mouse users prefer it. Try both and compare.

The mode is chosen **at calibration time** and stored inside `calibration.json`;
`run` automatically uses whatever mode you calibrated with.

### Click methods

- **`off`** (default) — cursor movement only.
- **`dwell`** — hold the cursor still (within 45 px) for 1 s → left click.
- **`blink`** — deliberately close both eyes for 0.25–1.5 s → left click.
  Normal reflex blinks (~0.1 s) are ignored.

---

## How it works

```
webcam frame (OpenCV)
   │
   ▼
MediaPipe FaceLandmarker  ──►  478 face landmarks incl. iris centers
   │
   ▼
feature extraction (tracking.py)
   • gaze_x/gaze_y : iris offset from eye-corner midpoint, ÷ eye width
   • yaw/pitch     : head pose from solvePnP on 6 rigid face points
   • ear           : eye aspect ratio (blink detector input)
   │
   ▼
calibration map (calibration.py)
   degree-2 polynomial ridge regression: features ─► screen (x, y) in [0,1]
   learned from you looking at 9 dots
   │
   ▼
One Euro filter (smoothing.py)
   adaptive smoothing: steady when you hold still, responsive when you flick
   │
   ▼
cursor (mouse.py) + click gestures (clicker.py)
```

### Why each piece exists

**Feature extraction.** Raw iris pixel positions are useless on their own —
they move when your head moves, when you lean closer, etc. So we measure the
iris *relative to the eye corners*, normalized by eye width (stable under
distance changes). Head pose (yaw/pitch) is included in gaze mode because
turning your head also shifts the iris in its socket; the regression needs
both signals to disentangle "eyes moved" from "head moved".

**Calibration.** There is no universal formula mapping eye features to screen
position — it depends on your screen size, sitting distance, and eye shape.
So we learn a per-user mapping: you look at 9 known points, we record the
median feature vector at each, then fit a small polynomial regression
(degree 2, ridge-regularized so 9 points can't overfit the quadratic terms).
The result is saved to `calibration.json`. **Recalibrate whenever you change
posture, chair height, or lighting significantly.**

**Smoothing.** Landmark jitter would make the raw cursor unusable. A One Euro
filter smooths adaptively: heavy filtering at low speed (rock-steady hover),
light filtering at high speed (no lag when you look across the screen). This
is the same filter class used in VR controllers.

**Clicking.** Dwell (hover-to-click) is the standard accessibility approach.
Blink uses the eye-aspect-ratio: below ~0.18 the eye is closed; a closure
lasting 0.25–1.5 s then reopening = intentional click.

---

## Project layout

```
touchless/
├── touchless/
│   ├── __main__.py     CLI (argparse) — start reading here
│   ├── app.py          main loops for preview / calibrate / run
│   ├── config.py       ALL tunable parameters, documented
│   ├── tracking.py     webcam + MediaPipe → FaceSample (the core contract)
│   ├── calibration.py  9-dot routine + polynomial ridge mapping
│   ├── smoothing.py    One Euro filter
│   ├── mouse.py        OS cursor wrapper (pyautogui)
│   └── clicker.py      dwell + blink click state machines
├── models/             auto-downloaded MediaPipe model (gitignored)
├── calibration.json    your personal calibration (gitignored)
└── requirements.txt
```

The key interface is `FaceSample` in `tracking.py`. Everything downstream
only knows about that dataclass — so when this becomes a Tauri app, you can
swap the perception layer (or move it behind a socket) without touching the
calibration/smoothing/click logic.

---

## Testing & iterating

### Verify tracking before anything else

```powershell
.\.venv\Scripts\python.exe -m touchless preview
```

You should see green dots on your eye corners and irises, and live numbers:

- `gaze x/y` should move when you look left/right/up/down **without moving
  your head** (x: roughly ±0.1 range; it's small — that's normal).
- `yaw/pitch` should track your head turning. (Absolute values may look odd —
  only *variation* matters; calibration absorbs offsets.)
- `ear` should drop below ~0.18 when you close your eyes. If it doesn't,
  tune `ear_closed_threshold` in `config.py`.

If tracking is jittery here, fix it *here* first (lighting, camera position)
— no amount of downstream tuning saves bad input. Face the light source;
avoid strong backlight.

### Evaluate accuracy after calibrating

Run with clicking off and try to "hit" things on screen with the cursor:

```powershell
.\.venv\Scripts\python.exe -m touchless run --click off
```

A quick objective test: open MS Paint fullscreen, dwell on each corner and
the center, see how tight the cursor cloud is at each. If accuracy is bad in
one screen region only → recalibrate. If it's bad everywhere → try `head`
mode, or sit closer to the camera.

### Tuning loop

All knobs live in `touchless/config.py` with comments. The ones you'll
actually touch:

| Symptom | Knob | Direction |
|---|---|---|
| Cursor jitters when holding still | `smooth_min_cutoff` | lower (e.g. 0.4) |
| Cursor lags behind fast glances | `smooth_beta` | higher (e.g. 1.0) |
| Dwell clicks fire accidentally | `dwell_time_s` / `dwell_radius_px` | raise / lower |
| Blinks not detected | `ear_closed_threshold` | raise (check `ear` in preview) |
| Calibration feels rushed | `calib_settle_s` | raise |

There are no unit tests yet — the honest test harness for this project is
`preview` (per-component signals) plus the Paint-corners drill (end-to-end).
A good next step: log `(feature, prediction)` pairs during `run` and build a
replay script so smoothing/mapping changes can be evaluated offline against
recorded sessions instead of your live face.

### Known limitations (MVP)

- Webcam gaze ≈ 9-region accuracy, not pixel accuracy. For fine targets you
  will want gaze-for-coarse + something else for fine (head micro-movements
  work well; that's the hybrid most real systems use).
- Single monitor assumed (uses primary screen size).
- Calibration is sensitive to posture changes — recalibration takes 25 s,
  use it liberally.
- Windows-focused (tested there); macOS needs accessibility permissions for
  pyautogui, Linux needs X11.

---

## Roadmap → Tauri

The plan this MVP feeds into:

1. **Finalize pipeline here** — mode choice (gaze vs head vs hybrid),
   click UX, smoothing constants. This is the hard, iterate-heavy part.
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
