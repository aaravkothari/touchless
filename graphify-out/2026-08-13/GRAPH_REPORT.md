# Graph Report - touchless  (2026-08-13)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 189 nodes · 411 edges · 11 communities
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 49 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `3f1ecc02`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- app.py
- _HandStack
- RidgePhysics
- FaceSample
- OneEuro2D
- calibration.py
- model.py
- Config
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

## Communities (11 total, 0 thin omitted)

### Community 0 - "app.py"
Cohesion: 0.11
Nodes (25): _draw_hud(), _face_hud_extras(), _Fps, _hand_points(), _hand_telemetry(), _load_model(), preview(), _preview_hand() (+17 more)

### Community 1 - "_HandStack"
Cohesion: 0.08
Nodes (21): _HandStack, Shared camera + hand landmarker every frame + face landmarker every Nth frame…, Camera, ndarray, Webcam capture, separated from the landmarkers so multiple trackers (face +…, ensure_hand_model(), HandSample, HandTracker (+13 more)

### Community 2 - "RidgePhysics"
Cohesion: 0.14
Nodes (9): AxisHGB, _expand(), ndarray, Small neural net on standardized features., (n, 8) raw features -> (n, 17) physics terms (bias, linear, tz*angles,…, The previous closed-form model, as a competing candidate., Gradient-boosted trees, one regressor per screen axis., RidgePhysics (+1 more)

### Community 3 - "FaceSample"
Cohesion: 0.18
Nodes (7): FaceSample, ndarray, Tongue-out detector. MediaPipe's blendshapes have no tongueOut, so: with the…, Grab a frame and extract features. Frame is unmirrored BGR., Run the landmarker on an externally captured BGR frame., Distance from camera, ~cm (the transformation matrix's -tz)., _tongue_score()

### Community 4 - "OneEuro2D"
Cohesion: 0.17
Nodes (6): _alpha(), _LowPass, OneEuro, OneEuro2D, One Euro filter — the standard choice for cursor-like signals. Plain moving…, Convenience wrapper: one filter per axis, shared parameters.

### Community 5 - "calibration.py"
Cohesion: 0.22
Nodes (12): _check_targets(), _collect_dot(), _instruction_screen(), ndarray, Calibration flow: pursuit collection -> model fit -> validation dots. The…, Pursuit collect -> fit -> validate -> user accepts or redoes. Returns the…, Mean of rows after MAD outlier rejection (falls back to median)., Validation dots: corners + center. (+4 more)

### Community 6 - "model.py"
Cohesion: 0.21
Nodes (7): build_dataset(), _candidates(), GazeModel, Learned gaze model: candidates compete on held-out data, best one wins.…, Apply hygiene filtering + pursuit-lag label shift. The eyes trail a moving…, Boolean holdout mask: the last `frac` of each phase, by time. Random splits…, _temporal_split()

### Community 7 - "Config"
Cohesion: 0.23
Nodes (8): calibrate(), Refit the model from the recorded pursuit session - no camera needed., retrain(), Config, Central tunables. Everything you'd want to tweak while iterating lives here., touchless — control the mouse cursor with your eyes/head via webcam., main(), CLI entry point: python -m touchless <command>.

### Community 8 - "Cursor"
Cohesion: 0.20
Nodes (3): Cursor, OS cursor control. Thin wrapper so the backend is swappable later. pyautogui's…, Move to normalized (0..1) screen coords, clamped inside the inset.

### Community 9 - "FaceTracker"
Cohesion: 0.22
Nodes (7): collect(), PursuitData, Pursuit calibration data collection: follow the moving cursor with your eyes.…, Run both pursuit phases. Returns collected data, or None on ESC., FaceTracker, Webcam capture + MediaPipe FaceLandmarker -> per-frame gaze/head features. This…, Face landmarker; owns the webcam unless one is shared in.

### Community 11 - "_Wander"
Cohesion: 0.47
Nodes (3): ndarray, Smooth random trajectory: eased travel between waypoints, random holds. Holds…, _Wander

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `app.py`, `_HandStack`, `RidgePhysics`, `FaceSample`, `calibration.py`, `model.py`, `FaceTracker`, `_Wander`?**
  _High betweenness centrality (0.369) - this node is a cross-community bridge._
- **Why does `FaceSample` connect `FaceSample` to `app.py`, `_HandStack`, `FaceTracker`, `Config`?**
  _High betweenness centrality (0.119) - this node is a cross-community bridge._
- **Why does `HandTracker` connect `_HandStack` to `app.py`, `Config`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `Config` (e.g. with `_Fps` and `_HandStack`) actually correct?**
  _`Config` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `FaceTracker` (e.g. with `_Fps` and `_HandStack`) actually correct?**
  _`FaceTracker` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `_Fps` (e.g. with `Camera` and `BlinkClicker`) actually correct?**
  _`_Fps` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `_HandStack` (e.g. with `Camera` and `BlinkClicker`) actually correct?**
  _`_HandStack` has 12 INFERRED edges - model-reasoned connections that need verification._