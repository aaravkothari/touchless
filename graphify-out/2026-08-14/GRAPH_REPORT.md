# Graph Report - touchless  (2026-08-14)

## Corpus Check
- 16 files · ~10,613 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 210 nodes · 430 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 49 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f4b42a9f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- app.py
- Camera
- PursuitData
- FaceSample
- OneEuro2D
- Config
- touchless
- CLAUDE.md
- Cursor
- FaceTracker
- _Wander

## God Nodes (most connected - your core abstractions)
1. `Config` - 49 edges
2. `FaceTracker` - 21 edges
3. `_Fps` - 19 edges
4. `_HandStack` - 19 edges
5. `FaceSample` - 18 edges
6. `Camera` - 17 edges
7. `PursuitData` - 17 edges
8. `run()` - 14 edges
9. `HandTracker` - 13 edges
10. `GazeModel` - 13 edges

## Surprising Connections (you probably didn't know these)
- `_Fps` --uses--> `Camera`  [INFERRED]
  touchless/app.py → touchless/camera.py
- `_Fps` --uses--> `Config`  [INFERRED]
  touchless/app.py → touchless/config.py
- `_Fps` --uses--> `HandTracker`  [INFERRED]
  touchless/app.py → touchless/hands.py
- `_Fps` --uses--> `GazeModel`  [INFERRED]
  touchless/app.py → touchless/model.py
- `_Fps` --uses--> `Cursor`  [INFERRED]
  touchless/app.py → touchless/mouse.py

## Import Cycles
- None detected.

## Communities (11 total, 2 thin omitted)

### Community 0 - "app.py"
Cohesion: 0.09
Nodes (27): _draw_hud(), _face_hud_extras(), _Fps, _hand_points(), _hand_telemetry(), _HandStack, _load_model(), preview() (+19 more)

### Community 1 - "Camera"
Cohesion: 0.11
Nodes (16): Camera, ndarray, Webcam capture, separated from the landmarkers so multiple trackers (face +…, ensure_hand_model(), HandSample, HandTracker, pinch_amount(), ndarray (+8 more)

### Community 2 - "PursuitData"
Cohesion: 0.09
Nodes (20): AxisHGB, build_dataset(), _candidates(), _expand(), GazeModel, ndarray, Learned gaze model: candidates compete on held-out data, best one wins.…, Small neural net on standardized features. (+12 more)

### Community 4 - "OneEuro2D"
Cohesion: 0.17
Nodes (6): _alpha(), _LowPass, OneEuro, OneEuro2D, One Euro filter — the standard choice for cursor-like signals. Plain moving…, Convenience wrapper: one filter per axis, shared parameters.

### Community 5 - "Config"
Cohesion: 0.13
Nodes (20): calibrate(), Refit the model from the recorded pursuit session - no camera needed., retrain(), _check_targets(), _collect_dot(), _instruction_screen(), ndarray, Calibration flow: pursuit collection -> model fit -> validation dots. The… (+12 more)

### Community 6 - "touchless"
Cohesion: 0.11
Nodes (18): 1. Verify tracking (`preview`), 2. Calibrate and read the numbers, 3. Iterate on the model without recollecting, Click methods, Commands, Hand mode, How it works, Known limitations (MVP) (+10 more)

### Community 8 - "Cursor"
Cohesion: 0.20
Nodes (3): Cursor, OS cursor control. Thin wrapper so the backend is swappable later. pyautogui's…, Move to normalized (0..1) screen coords, clamped inside the inset.

### Community 9 - "FaceTracker"
Cohesion: 0.17
Nodes (11): ensure_model(), FaceTracker, ndarray, Path, Webcam capture + MediaPipe FaceLandmarker -> per-frame gaze/head features. This…, Tongue-out detector. MediaPipe's blendshapes have no tongueOut, so: with the…, Face landmarker; owns the webcam unless one is shared in., Grab a frame and extract features. Frame is unmirrored BGR. (+3 more)

### Community 11 - "_Wander"
Cohesion: 0.47
Nodes (3): ndarray, Smooth random trajectory: eased travel between waypoints, random holds. Holds…, _Wander

## Knowledge Gaps
- **14 isolated node(s):** `graphify`, `Safety / emergency stop`, `Click methods`, `Hand mode`, `Pursuit calibration (calibration.py + pursuit.py)` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `app.py`, `Camera`, `PursuitData`, `FaceSample`, `FaceTracker`, `_Wander`?**
  _High betweenness centrality (0.298) - this node is a cross-community bridge._
- **Why does `FaceSample` connect `FaceSample` to `app.py`, `Camera`, `Config`, `FaceTracker`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `HandTracker` connect `Camera` to `app.py`, `Config`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `Config` (e.g. with `_Fps` and `_HandStack`) actually correct?**
  _`Config` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `FaceTracker` (e.g. with `_Fps` and `_HandStack`) actually correct?**
  _`FaceTracker` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `_Fps` (e.g. with `Camera` and `BlinkClicker`) actually correct?**
  _`_Fps` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `_HandStack` (e.g. with `Camera` and `BlinkClicker`) actually correct?**
  _`_HandStack` has 12 INFERRED edges - model-reasoned connections that need verification._