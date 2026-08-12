# touchless

Control your mouse cursor with your eyes + head using a plain webcam.

This is the Python MVP of a project that will eventually become a Tauri app.
The goal of this stage is to **finalize the tracking/control pipeline** —
calibration, the learned gaze model, smoothing, click gestures — before any
app-shell work.

---

## Quick start

```powershell
# one-time setup
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 1. sanity-check your camera and lighting (no cursor control)
.\.venv\Scripts\python.exe -m touchless preview

# 2. calibrate: follow the moving cursor with your eyes (~2.5 min)
.\.venv\Scripts\python.exe -m touchless calibrate

# 3. drive the cursor
.\.venv\Scripts\python.exe -m touchless run

# or skip calibration entirely and point with your finger instead
.\.venv\Scripts\python.exe -m touchless run --input hand
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
| `python -m touchless calibrate` | Pursuit calibration: you follow the real moving mouse cursor for two ~60 s phases, three ML models compete on your data, then static check dots measure real accuracy in px before you save (`Enter`), redo (`r`), or abort (`Esc`). |
| `python -m touchless retrain` | Refit the model from the last recorded session (`gaze_data.npz`) — no camera needed. For iterating on model code offline. |
| `python -m touchless run [--input gaze\|hand] [--click ...] [--log FILE.csv]` | Drive the cursor. `gaze` (default) uses the trained model; `hand` follows your right index finger with no calibration needed. |
| `--camera N` (any command) | Use webcam index N if you have more than one. |

### Click methods

- **`off`** (default) — cursor movement only.
- **`dwell`** — hold the cursor still (within 45 px) for 1 s → left click. Works in both input modes.
- **`blink`** — deliberately close both eyes for 0.25–1.5 s → left click (gaze mode only).
  Reflex blinks (~0.1 s) are ignored.
- Hand mode ignores `--click`: its control scheme is fixed (left-hand pinches
  click and drag, tongue recenters — see the Hand mode section).

### Hand mode

```powershell
.\.venv\Scripts\python.exe -m touchless preview --input hand   # check detection first
.\.venv\Scripts\python.exe -m touchless run --input hand
```

An "air trackpad" with a fixed control scheme — **right hand moves, left
hand + face control** (`--click` is ignored in this mode):

| Gesture | Action |
|---|---|
| Point with **right index finger** | Move the cursor (`cursor = center + gain × (finger − anchor)`; absolute, no drift, no training) |
| **Left hand**: pinch thumb + index | **Left click** — hold the pinch to hold the button (drag!) |
| **Left hand**: pinch thumb + middle | **Right click** — hold works the same |
| **Tongue out** (~0.2 s) | Recenter: cursor snaps to screen center and your current finger position becomes the new neutral |
| Hand leaves the frame | Cursor holds still; any held button releases |

Notes:

- Tongue detection is a custom heuristic (MediaPipe's blendshapes have no
  `tongueOut`): with your jaw open, the app checks whether the inner-mouth
  region is filled with bright tongue-colored pixels instead of a dark
  cavity. Watch the `FACE jaw/tongue` numbers in preview — tongue should
  jump above ~0.45 when you stick it out. Lighting-sensitive; tune
  `tongue_threshold` if needed.
- Pinch-hold uses hysteresis (engage < 0.35, release > 0.45 of hand size)
  so drags don't flutter.
- Hand mode runs the face model too (for the tongue), at half frame rate
  (`face_every_n`) — expect ~20 fps overall.
- If preview labels your hands wrong, flip `hand_labels_flipped` in config.

---

## How it works

```
webcam frame, 1280x720 (OpenCV)
   │
   ▼
MediaPipe FaceLandmarker (tracking.py)
   blendshapes ─► gaze coefficients + blink;  transform matrix ─► yaw/pitch/roll + x/y/depth
   │
   ▼
feature vector: [gaze_x, gaze_y, yaw, pitch, roll, tx, ty, tz]
   │
   ▼
learned gaze model (model.py — trained by pursuit calibration)
   ridge-physics vs MLP vs gradient boosting; best holdout error wins
   │
   ▼
robustness (app.py): blink gate → clamp → median-of-3 → One Euro filter
   │
   ▼
cursor (mouse.py) + click gestures (clicker.py)
```

### Pursuit calibration (calibration.py + pursuit.py)

You **follow the actual mouse cursor** as it wanders smoothly around the
screen — eased travel between random waypoints, brief holds, forced visits
to edges and corners. Every frame records (face features → where the cursor
is). Two phases:

1. ~60 s sitting comfortably.
2. ~60 s **while slowly changing posture** — lean back, lean in, shift
   around. Position invariance is learned from continuous real data, not
   assumed from a formula.

That's ~3,500 training samples instead of the ~34 static dots of the
previous design — enough data to train actual ML models.

### The model shootout (model.py)

Every calibration trains three candidates and prints a comparison:

```
pursuit lag: 100 ms (ridge holdout 62px)
dataset: 3135 samples (627 held out)
model comparison (holdout, lower is better):
  hgb        58px   (fit 3.3s)
  mlp        64px   (fit 4.0s)
  ridge      71px   (fit 0.0s)
winner: hgb (58px holdout), refit on all data
```

- **ridge** — regression on a physics-motivated term expansion (the old
  closed-form approach, kept as the baseline that keeps us honest)
- **mlp** — small neural net (2×64, standardized inputs, early stopping)
- **hgb** — gradient-boosted trees, one per screen axis

Details that matter:

- **Pursuit lag is measured, not assumed.** Your eyes trail a moving target
  by ~100 ms, so features at time *t* are paired with where the target was
  *lag* seconds earlier. The lag (0–250 ms) is chosen by holdout error.
- **Temporal holdout, not random.** The last 20% of each phase is held out.
  Random splits would leak — adjacent frames are near-duplicates.
- **Hygiene:** frames during a blink (gaze coefficients go haywire) and for
  0.2 s afterwards are dropped before training.
- The winner is refit on all data, then verified on **static check dots**
  (5 at normal posture + 1 leaned back) and you see the real px errors
  before anything is saved.

### Runtime robustness (app.py)

While you blink the cursor holds still. Raw predictions are clamped, then
median-of-3 filtered (kills single-frame spikes), then One Euro smoothed
(steady when you hold still, responsive when you flick).

---

## Project layout

```
touchless/
├── touchless/
│   ├── __main__.py     CLI (argparse) — start reading here
│   ├── app.py          main loops: preview / calibrate / retrain / run
│   ├── config.py       ALL tunable parameters, documented
│   ├── tracking.py     webcam + MediaPipe → FaceSample (the core contract)
│   ├── hands.py        webcam + MediaPipe → HandSample (hand mode perception)
│   ├── pursuit.py      moving-cursor trajectory + data collection
│   ├── model.py        dataset build, lag search, model shootout, persistence
│   ├── calibration.py  calibration UX: instructions, validation dots, accept/redo
│   ├── smoothing.py    One Euro filter
│   ├── mouse.py        OS cursor wrapper (pyautogui)
│   └── clicker.py      dwell + blink click state machines
├── models/             auto-downloaded MediaPipe model (gitignored)
├── model.pkl           your trained gaze model (gitignored, versioned)
├── gaze_data.npz       your last pursuit session (gitignored) — feeds `retrain`
└── requirements.txt
```

The key interface is `FaceSample` in `tracking.py` plus
`GazeModel.predict(features) -> (x, y)` in `model.py`. Everything else is
replaceable — including the model itself, which is the point.

---

## Testing & iterating

### 1. Verify tracking (`preview`)

The telemetry panel shows everything being tracked, live: gaze x/y, blink
score, yaw/pitch/roll, head x/y in cm, and **depth in cm with a bar that
moves as you lean**. Check each signal responds before calibrating. fps
should be ≥ 20; if not, drop `frame_width/height` in `config.py`.

### 2. Calibrate and read the numbers

During phase 2, genuinely move — lean back as far as you ever sit, lean in
close, shift side to side. The model can only be accurate in postures it
has seen. The console prints the lag, dataset size, and model comparison;
the check screen shows mean/worst error at normal posture and leaned back.
Rough guide on a 1920 screen: **< 100 px mean is good** for a webcam,
~150 px is usable with dwell clicking.

### 3. Iterate on the model without recollecting

`gaze_data.npz` holds your raw session. Edit `model.py` (add a candidate,
change hyperparameters, widen the lag grid) and run:

```powershell
.\.venv\Scripts\python.exe -m touchless retrain
```

Instant experiment loop against your own recorded face — no camera, no
recalibration. This is the main iteration surface now.

### Tuning loop

| Symptom | Knob | Direction |
|---|---|---|
| Cursor jitters when holding still | `smooth_min_cutoff` | lower (e.g. 0.3) |
| Cursor lags behind fast glances | `smooth_beta` | higher (e.g. 1.0) |
| Cursor freezes too often (squint reads as blink) | `blink_gate` | raise |
| Dot moves too fast to track comfortably | `pursuit_speed` | lower the max |
| Model overfits phase 1 posture | move MORE during phase 2 | — |
| Dwell clicks fire accidentally | `dwell_time_s` / `dwell_radius_px` | raise / lower |
| Blink-clicks not detected | `blink_closed_threshold` | lower |
| Hand cursor too twitchy / too sluggish | `hand_gain` | lower / raise |
| Wrong hand labeled RIGHT in hand preview | `hand_labels_flipped` | flip |
| Pinch clicks fire accidentally | `pinch_click_threshold` | lower |
| Drags drop mid-hold | `pinch_release_threshold` | raise |
| Tongue recenter won't trigger | `tongue_threshold` | lower (watch `tongue` in preview) |
| Talking triggers recenter | `tongue_threshold` / `tongue_hold_s` | raise |

### Known limitations (MVP)

- Physics still caps webcam gaze at region-level precision — the learned
  model squeezes out what the signal contains, it cannot add signal. Head
  movement gives the fine control; eyes give fast coarse targeting.
- Sessions overwrite `gaze_data.npz`; appending multiple sessions for a
  bigger training set is an easy future upgrade.
- Single monitor assumed; Windows-tested.

---

## Roadmap → Tauri

1. **Finalize pipeline here** — accuracy, click UX, smoothing constants.
   *(you are here)*
2. **Split engine from UI** — run the tracker as a headless process emitting
   `(x, y, click)` over a local WebSocket.
3. **Tauri app** — frontend = settings/calibration UI; engine = PyInstaller
   sidecar (fast path) or Rust rewrite with ONNX Runtime (clean path).
   Note: the trained model is a pickled sklearn object; the Rust path would
   export it (or retrain a numpy-portable model) — one more reason the
   engine/UI split comes first.
