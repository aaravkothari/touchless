"""Webcam capture + MediaPipe FaceLandmarker -> per-frame gaze/head features.

This is the perception layer. Everything downstream (calibration, cursor
mapping) consumes the FaceSample produced here, so if you swap the tracking
backend later (e.g. an ONNX model in Rust for the Tauri port) this is the
only contract that matters.

Uses the MediaPipe Tasks API (mediapipe >= 1.0 removed the old
mp.solutions.face_mesh API). The model file is downloaded automatically on
first run into models/.
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

# FaceLandmarker landmark indices (478 points; 468-477 are the irises).
# "Right"/"left" are the subject's right/left, which appear on the image's
# left/right respectively when the frame is not mirrored.
R_OUTER, R_INNER, R_TOP, R_BOTTOM, R_IRIS = 33, 133, 159, 145, 468
L_INNER, L_OUTER, L_TOP, L_BOTTOM, L_IRIS = 362, 263, 386, 374, 473

# Landmarks used for head-pose estimation via solvePnP.
POSE_IDS = [1, 152, 263, 33, 291, 61]  # nose, chin, eye corners, mouth corners

# Generic 3D face model points matching POSE_IDS (millimetres, arbitrary origin).
POSE_MODEL = np.array([
    [0.0, 0.0, 0.0],        # nose tip
    [0.0, -63.6, -12.5],    # chin
    [-43.3, 32.7, -26.0],   # left eye outer corner
    [43.3, 32.7, -26.0],    # right eye outer corner
    [-28.9, -28.9, -24.1],  # left mouth corner
    [28.9, -28.9, -24.1],   # right mouth corner
], dtype=np.float64)


@dataclass
class FaceSample:
    ok: bool = False
    gaze_x: float = 0.0   # iris offset within the eyes, normalized, + = subject's left
    gaze_y: float = 0.0   # + = down
    yaw: float = 0.0      # head rotation, degrees, + = subject looks to their left
    pitch: float = 0.0    # degrees, + = looks down
    ear: float = 0.0      # eye aspect ratio, averaged over both eyes (blink = low)
    landmarks: np.ndarray | None = field(default=None, repr=False)  # (478, 2) px


def ensure_model() -> Path:
    """Download the FaceLandmarker model on first run (~3.7 MB)."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


def _eye_features(pts: np.ndarray, outer: int, inner: int, top: int,
                  bottom: int, iris: int) -> tuple[float, float, float]:
    """Return (iris_dx, iris_dy, ear) for one eye.

    The iris offset is measured from the eye-corner midpoint and normalized
    by eye width, which is stable under distance changes and (unlike eye
    height) doesn't collapse when the lids move.
    """
    width = float(np.linalg.norm(pts[outer] - pts[inner])) + 1e-9
    center = (pts[outer] + pts[inner]) / 2.0
    dx, dy = (pts[iris] - center) / width
    ear = float(np.linalg.norm(pts[top] - pts[bottom])) / width
    return float(dx), float(dy), ear


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
        if not result.face_landmarks:
            return frame, FaceSample()

        h, w = frame.shape[:2]
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]],
            dtype=np.float64,
        )

        r_dx, r_dy, r_ear = _eye_features(pts, R_OUTER, R_INNER, R_TOP, R_BOTTOM, R_IRIS)
        l_dx, l_dy, l_ear = _eye_features(pts, L_OUTER, L_INNER, L_TOP, L_BOTTOM, L_IRIS)
        yaw, pitch = self._head_pose(pts, w, h)

        return frame, FaceSample(
            ok=True,
            gaze_x=(r_dx + l_dx) / 2.0,
            gaze_y=(r_dy + l_dy) / 2.0,
            yaw=yaw,
            pitch=pitch,
            ear=(r_ear + l_ear) / 2.0,
            landmarks=pts,
        )

    def _head_pose(self, pts: np.ndarray, w: int, h: int) -> tuple[float, float]:
        """Estimate (yaw, pitch) in degrees with solvePnP and a generic face model."""
        image_pts = pts[POSE_IDS]
        focal = float(w)  # rough pinhole approximation, fine for relative pose
        cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            POSE_MODEL, image_pts, cam, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return 0.0, 0.0
        rot, _ = cv2.Rodrigues(rvec)
        # Decompose: yaw around Y, pitch around X.
        yaw = float(np.degrees(np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))))
        pitch = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
        return yaw, pitch

    def close(self):
        self.landmarker.close()
        self.cap.release()
