"""Webcam capture + MediaPipe HandLandmarker -> per-frame hand state.

Perception layer for hand mode: the user's right index fingertip drives the
cursor, a left-hand fist recenters it, and a right-hand thumb-index pinch
can click. Same module shape as tracking.py (the face equivalent).
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
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

# Hand landmark indices (21 per hand).
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
_TIPS = (8, 12, 16, 20)   # index..pinky fingertips
_PIPS = (6, 10, 14, 18)   # matching middle joints


@dataclass
class HandSample:
    pointer_ok: bool = False
    pointer: np.ndarray | None = None   # (2,) right index tip, normalized cam coords
    pinch: float = 9.9                  # thumb-index distance / hand size (right hand)
    left_ok: bool = False
    left_fist: bool = False
    hands: list = field(default_factory=list, repr=False)  # [(label, (21,2) px)]


def is_fist(lm: np.ndarray) -> bool:
    """Fist = fingertips folded in: tip closer to the wrist than its middle
    joint for at least 3 of 4 fingers (thumb excluded, it's unreliable)."""
    wrist = lm[WRIST]
    folded = sum(
        np.linalg.norm(lm[tip] - wrist) < np.linalg.norm(lm[pip] - wrist)
        for tip, pip in zip(_TIPS, _PIPS)
    )
    return folded >= 3


def pinch_amount(lm: np.ndarray) -> float:
    """Thumb-tip to index-tip distance, normalized by hand size."""
    size = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST])) + 1e-9
    return float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP])) / size


def ensure_hand_model() -> Path:
    """Download the HandLandmarker model on first run (~7.5 MB)."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


class HandTracker:
    """Owns the webcam and the landmarker; yields one HandSample per frame."""

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
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(ensure_hand_model())),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    def _user_hand(self, label: str) -> str:
        """MediaPipe predicts handedness assuming a mirrored (selfie) image;
        our frames are unmirrored, so the reported label is the opposite of
        the user's actual hand. hand_labels_flipped=False turns this off if
        a camera pipeline mirrors for you."""
        if self.cfg.hand_labels_flipped:
            return "Right" if label == "Left" else "Left"
        return label

    def read(self) -> tuple[np.ndarray | None, HandSample]:
        """Grab a frame and extract hand state. Frame is unmirrored BGR."""
        ok, frame = self.cap.read()
        if not ok:
            return None, HandSample()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int((time.monotonic() - self._t0) * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        sample = HandSample()
        h, w = frame.shape[:2]
        for lm_list, handed in zip(result.hand_landmarks, result.handedness):
            lm = np.array([(p.x, p.y) for p in lm_list], dtype=np.float64)
            label = self._user_hand(handed[0].category_name)
            sample.hands.append((label, lm * np.array([w, h])))
            if label == "Right":
                sample.pointer_ok = True
                sample.pointer = lm[INDEX_TIP].copy()
                sample.pinch = pinch_amount(lm)
            elif label == "Left":
                sample.left_ok = True
                sample.left_fist = is_fist(lm)
        return frame, sample

    def close(self):
        self.landmarker.close()
        self.cap.release()
