# Graph Report - touchless  (2026-08-15)

## Corpus Check
- 18 files · ~13,215 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 236 nodes · 478 edges · 10 communities (9 shown, 1 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 53 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `83b16788`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- app.py
- Camera
- model.py
- FaceSample
- OneEuro2D
- Config
- touchless
- CLAUDE.md
- Cursor
- HandPointerPipeline

## God Nodes (most connected - your core abstractions)
1. `Config` - 54 edges
2. `FaceTracker` - 21 edges
3. `_Fps` - 20 edges
4. `_HandStack` - 20 edges
5. `FaceSample` - 18 edges
6. `Camera` - 17 edges
7. `PursuitData` - 17 edges
8. `HandPointerPipeline` - 15 edges
9. `run()` - 14 edges
10. `HandTracker` - 13 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `Config`  [EXTRACTED]
  scripts/sim_stability.py → touchless/config.py
- `run()` --calls--> `HandPointerPipeline`  [EXTRACTED]
  scripts/sim_stability.py → touchless/pipeline.py
- `_Fps` --uses--> `Camera`  [INFERRED]
  touchless/app.py → touchless/camera.py
- `_Fps` --uses--> `Config`  [INFERRED]
  touchless/app.py → touchless/config.py
- `_Fps` --uses--> `HandTracker`  [INFERRED]
  touchless/app.py → touchless/hands.py

## Import Cycles
- None detected.

## Communities (10 total, 1 thin omitted)

### Community 0 - "app.py"
Cohesion: 0.09
Nodes (28): _draw_hud(), _face_hud_extras(), _Fps, _hand_points(), _hand_telemetry(), _HandStack, _load_model(), preview() (+20 more)

### Community 1 - "Camera"
Cohesion: 0.09
Nodes (20): Camera, ndarray, Webcam capture, separated from the landmarkers so multiple trackers (face +…, ensure_hand_model(), HandSample, HandTracker, pinch_amount(), ndarray (+12 more)

### Community 2 - "model.py"
Cohesion: 0.12
Nodes (13): AxisHGB, _candidates(), _expand(), ndarray, Learned gaze model: candidates compete on held-out data, best one wins.…, Small neural net on standardized features., Boolean holdout mask: the last `frac` of each phase, by time. Random splits…, (n, 8) raw features -> (n, 17) physics terms (bias, linear, tz*angles,… (+5 more)

### Community 3 - "FaceSample"
Cohesion: 0.18
Nodes (7): FaceSample, ndarray, Tongue-out detector. MediaPipe's blendshapes have no tongueOut, so: with the…, Grab a frame and extract features. Frame is unmirrored BGR., Run the landmarker on an externally captured BGR frame., Distance from camera, ~cm (the transformation matrix's -tz)., _tongue_score()

### Community 4 - "OneEuro2D"
Cohesion: 0.16
Nodes (6): _alpha(), _LowPass, OneEuro, OneEuro2D, One Euro filter — the standard choice for cursor-like signals. Plain moving…, Convenience wrapper: one filter per axis, shared parameters. d_cutoff low-…

### Community 5 - "Config"
Cohesion: 0.09
Nodes (30): calibrate(), Refit the model from the recorded pursuit session - no camera needed., retrain(), _check_targets(), _collect_dot(), _instruction_screen(), ndarray, Calibration flow: pursuit collection -> model fit -> validation dots. The… (+22 more)

### Community 6 - "touchless"
Cohesion: 0.11
Nodes (18): 1. Verify tracking (`preview`), 2. Calibrate and read the numbers, 3. Iterate on the model without recollecting, Click methods, Commands, Hand mode, How it works, Known limitations (MVP) (+10 more)

### Community 8 - "Cursor"
Cohesion: 0.20
Nodes (3): Cursor, OS cursor control. Thin wrapper so the backend is swappable later. pyautogui's…, Move to normalized (0..1) screen coords, clamped inside the inset.

### Community 9 - "HandPointerPipeline"
Cohesion: 0.10
Nodes (20): main(), make_track(), Offline stability harness for the hand pointer pipeline. Feeds…, Seconds from move start until the cursor stays within 15 px of its final still2…, % of a slow deliberate move that actually reached the cursor (the gate must not…, (t, pointer, phase) arrays for one synthetic session., report(), run() (+12 more)

## Knowledge Gaps
- **14 isolated node(s):** `graphify`, `Safety / emergency stop`, `Click methods`, `Hand mode`, `Pursuit calibration (calibration.py + pursuit.py)` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `app.py`, `Camera`, `model.py`, `FaceSample`, `OneEuro2D`, `HandPointerPipeline`?**
  _High betweenness centrality (0.345) - this node is a cross-community bridge._
- **Why does `FaceSample` connect `FaceSample` to `app.py`, `Camera`, `Config`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `HandPointerPipeline` connect `HandPointerPipeline` to `app.py`, `OneEuro2D`, `Config`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `Config` (e.g. with `_Fps` and `_HandStack`) actually correct?**
  _`Config` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `FaceTracker` (e.g. with `_Fps` and `_HandStack`) actually correct?**
  _`FaceTracker` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `_Fps` (e.g. with `Camera` and `BlinkClicker`) actually correct?**
  _`_Fps` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `_HandStack` (e.g. with `Camera` and `BlinkClicker`) actually correct?**
  _`_HandStack` has 13 INFERRED edges - model-reasoned connections that need verification._