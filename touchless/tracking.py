"""Webcam capture + MediaPipe FaceLandmarker -> per-frame gaze/head features.

This is the perception layer. Everything downstream (calibration, cursor
mapping) consumes the FaceSample produced here, so if you swap the tracking
backend later (e.g. an ONNX model in Rust for the Tauri port) this is the
only contract that matters.

Features come from the model itself, not hand-rolled geometry:
  - gaze:      the 8 eyeLook* blendshapes (trained gaze coefficients)
  - head pose: yaw/pitch + translation from the facial transformation matrix
  - blink:     the eyeBlink* blendshapes
"""

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .config import Config

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# Order matters: this is the feature vector calibration learns over.
FEATURE_NAMES = ("gaze_x", "gaze_y", "yaw", "pitch", "roll", "tx", "ty", "tz")


@dataclass
class FaceSample:
    ok: bool = False
    features: np.ndarray | None = field(default=None)  # (8,) see FEATURE_NAMES
    blink: float = 0.0    # mean eyeBlink blendshape, 0 (open) .. 1 (closed)
    landmarks: np.ndarray | None = field(default=None, repr=False)  # (478, 2) px

    def _f(self, i: int) -> float:
        return float(self.features[i]) if self.features is not None else 0.0

    @property
    def gaze_x(self) -> float:
        return self._f(0)

    @property
    def gaze_y(self) -> float:
        return self._f(1)

    @property
    def yaw(self) -> float:
        return self._f(2)

    @property
    def pitch(self) -> float:
        return self._f(3)

    @property
    def roll(self) -> float:
        return self._f(4)

    @property
    def tx(self) -> float:
        return self._f(5)

    @property
    def ty(self) -> float:
        return self._f(6)

    @property
    def depth(self) -> float:
        """Distance from camera, ~cm (the transformation matrix's -tz)."""
        return self._f(7)


def ensure_model() -> Path:
    """Download the FaceLandmarker model on first run (~3.7 MB)."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


class FaceTracker:
    """Owns the webcam and the landmarker; yields one FaceSample per frame."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cfg.camera_index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {cfg.camera_index}. "
                "Try a different --camera index, and check no other app is using it."
            )
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(ensure_model())),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    def read(self) -> tuple[np.ndarray | None, FaceSample]:
        """Grab a frame and extract features. Frame is unmirrored BGR."""
        ok, frame = self.cap.read()
        if not ok:
            return None, FaceSample()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps.
        ts_ms = max(int((time.monotonic() - self._t0) * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not (result.face_landmarks and result.face_blendshapes
                and result.facial_transformation_matrixes):
            return frame, FaceSample()

        h, w = frame.shape[:2]
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]],
            dtype=np.float64,
        )
        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}

        # Gaze: + gaze_x = subject looks to their left, + gaze_y = looks down.
        # (Sign conventions don't actually matter — calibration learns the map —
        # but consistent semantics make the preview HUD interpretable.)
        gaze_x = (bs["eyeLookOutLeft"] + bs["eyeLookInRight"]
                  - bs["eyeLookInLeft"] - bs["eyeLookOutRight"]) / 2.0
        gaze_y = (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"]
                  - bs["eyeLookUpLeft"] - bs["eyeLookUpRight"]) / 2.0

        # Head pose from the facial transformation matrix (stable, model-learned).
        M = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
        R = M[:3, :3]
        yaw = float(np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))))
        pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        tx, ty = float(M[0, 3]), float(M[1, 3])  # head translation, ~cm
        tz = float(-M[2, 3])                     # distance from camera, ~cm (+)

        return frame, FaceSample(
            ok=True,
            features=np.array([gaze_x, gaze_y, yaw, pitch, roll, tx, ty, tz]),
            blink=(bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2.0,
            landmarks=pts,
        )

    def close(self):
        self.landmarker.close()
        self.cap.release()
