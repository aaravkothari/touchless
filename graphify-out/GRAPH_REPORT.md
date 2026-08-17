# Graph Report - touchless  (2026-08-16)

## Corpus Check
- 18 files · ~16,655 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 262 nodes · 528 edges · 11 communities (10 shown, 1 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 57 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `64d2de94`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Config
- _HandStack
- model.py
- FaceSample
- _Fps
- FaceTracker
- touchless
- CLAUDE.md
- Cursor
- sim_stability.py
- ArmGate

## God Nodes (most connected - your core abstractions)
1. `Config` - 56 edges
2. `_Fps` - 21 edges
3. `_HandStack` - 21 edges
4. `FaceTracker` - 21 edges
5. `ArmGate` - 18 edges
6. `FaceSample` - 18 edges
7. `Camera` - 17 edges
8. `PursuitData` - 17 edges
9. `HandPointerPipeline` - 16 edges
10. `run()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `run()` --calls--> `HandPointerPipeline`  [EXTRACTED]
  scripts/sim_stability.py → touchless/pipeline.py
- `run_pure()` --calls--> `ArmGate`  [EXTRACTED]
  scripts/sim_stability.py → touchless/pipeline.py
- `run_pure()` --calls--> `HandPointerPipeline`  [EXTRACTED]
  scripts/sim_stability.py → touchless/pipeline.py
- `main()` --calls--> `Config`  [EXTRACTED]
  scripts/sim_stability.py → touchless/config.py
- `_Fps` --uses--> `Camera`  [INFERRED]
  touchless/app.py → touchless/camera.py

## Import Cycles
- None detected.

## Communities (11 total, 1 thin omitted)

### Community 0 - "Config"
Cohesion: 0.08
Nodes (34): calibrate(), _draw_hud(), _face_hud_extras(), _hand_points(), _hand_telemetry(), _load_model(), preview(), _preview_hand() (+26 more)

### Community 1 - "_HandStack"
Cohesion: 0.09
Nodes (19): _HandStack, Shared camera + hand landmarker every frame + face landmarker every Nth frame…, Camera, ndarray, ensure_hand_model(), HandSample, HandTracker, pinch_amount() (+11 more)

### Community 2 - "model.py"
Cohesion: 0.12
Nodes (13): AxisHGB, _candidates(), _expand(), ndarray, Learned gaze model: candidates compete on held-out data, best one wins.…, Small neural net on standardized features., Boolean holdout mask: the last `frac` of each phase, by time. Random splits…, (n, 8) raw features -> (n, 17) physics terms (bias, linear, tz*angles,… (+5 more)

### Community 3 - "FaceSample"
Cohesion: 0.18
Nodes (7): FaceSample, ndarray, Tongue-out detector. MediaPipe's blendshapes have no tongueOut, so: with the…, Grab a frame and extract features. Frame is unmirrored BGR., Run the landmarker on an externally captured BGR frame., Distance from camera, ~cm (the transformation matrix's -tz)., _tongue_score()

### Community 4 - "_Fps"
Cohesion: 0.09
Nodes (13): _Fps, HandPointerPipeline, Hand-mode pointer -> cursor pipeline, extracted for testability. The transform…, pointer (camera / hand-size units) -> normalized screen target. Stages:…, Pointer left the frame: hold position, forget stillness state., Pause toggled: drop all filter state so resume starts clean., [STILL] HUD line for threshold tuning., _alpha() (+5 more)

### Community 5 - "FaceTracker"
Cohesion: 0.09
Nodes (25): _check_targets(), _collect_dot(), _instruction_screen(), ndarray, Calibration flow: pursuit collection -> model fit -> validation dots. The…, Pursuit collect -> fit -> validate -> user accepts or redoes. Returns the…, Mean of rows after MAD outlier rejection (falls back to median)., Validation dots: corners + center. (+17 more)

### Community 6 - "touchless"
Cohesion: 0.11
Nodes (18): 1. Verify tracking (`preview`), 2. Calibrate and read the numbers, 3. Iterate on the model without recollecting, Click methods, Commands, Hand mode, How it works, Known limitations (MVP) (+10 more)

### Community 8 - "Cursor"
Cohesion: 0.20
Nodes (3): Cursor, OS cursor control. Thin wrapper so the backend is swappable later. pyautogui's…, Move to normalized (0..1) screen coords, clamped inside the inset.

### Community 9 - "sim_stability.py"
Cohesion: 0.14
Nodes (23): detect_lag_frames(), finger_track_pct(), leak_px(), main(), make_track(), make_track_pure(), _phase_end_px(), Offline stability harness for the hand pointer pipeline. Feeds… (+15 more)

### Community 10 - "ArmGate"
Cohesion: 0.14
Nodes (9): ArmGate, ndarray, (tip, ref) -> virtual pointer that ignores whole-arm translation. Hand-pure…, First sample at or after max(now-span, floor) (newest sample before it if the…, Mean of samples in [t0, t1] (None if the span is empty)., Right hand left the frame: hold the virtual, remember the last tip so the…, Pause toggled: same handling as a lost pointer., [ARM] HUD line for threshold tuning. (+1 more)

## Knowledge Gaps
- **14 isolated node(s):** `graphify`, `Safety / emergency stop`, `Click methods`, `Hand mode`, `Pursuit calibration (calibration.py + pursuit.py)` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Config` connect `Config` to `_HandStack`, `model.py`, `FaceSample`, `_Fps`, `FaceTracker`, `sim_stability.py`, `ArmGate`?**
  _High betweenness centrality (0.362) - this node is a cross-community bridge._
- **Why does `ArmGate` connect `ArmGate` to `Config`, `sim_stability.py`, `_Fps`, `_HandStack`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Why does `FaceSample` connect `FaceSample` to `Config`, `_HandStack`, `_Fps`, `FaceTracker`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `Config` (e.g. with `_Fps` and `_HandStack`) actually correct?**
  _`Config` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `_Fps` (e.g. with `Camera` and `BlinkClicker`) actually correct?**
  _`_Fps` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `_HandStack` (e.g. with `Camera` and `BlinkClicker`) actually correct?**
  _`_HandStack` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `FaceTracker` (e.g. with `_Fps` and `_HandStack`) actually correct?**
  _`FaceTracker` has 6 INFERRED edges - model-reasoned connections that need verification._