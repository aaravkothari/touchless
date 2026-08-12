"""Webcam capture + MediaPipe HandLandmarker -> per-frame hand state.

Control split for hand mode: the RIGHT hand only moves the cursor (index
fingertip); the LEFT hand is the button box — thumb+index pinch = left
click, thumb+middle pinch = right click, holding a pinch holds the button.
Same module shape as tracking.py (the face equivalent).
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

from .camera import Camera
from .config import Config

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

# Hand landmark indices (21 per hand).
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP = 0, 4, 8, 9, 12


@dataclass
class HandSample:
    pointer_ok: bool = False
    pointer: np.ndarray | None = None   # (2,) right index tip, normalized cam coords
    left_ok: bool = False
    pinch_index: float = 9.9            # left thumb<->index dist / hand size
    pinch_middle: float = 9.9           # left thumb<->middle dist / hand size
    hands: list = field(default_factory=list, repr=False)  # [(label, (21,2) px)]


def pinch_amount(lm: np.ndarray, tip: int) -> float:
    """Thumb-tip to given fingertip distance, normalized by hand size."""
    size = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST])) + 1e-9
    return float(np.linalg.norm(lm[THUMB_TIP] - lm[tip])) / size


def ensure_hand_model() -> Path:
    """Download the HandLandmarker model on first run (~7.5 MB)."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


class HandTracker:
    """Hand landmarker; owns the webcam unless one is shared in."""

    def __init__(self, cfg: Config, camera: Camera | None = None):
        self.cfg = cfg
        self._own_camera = camera is None
        self.camera = camera if camera is not None else Camera(cfg)
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
        """Which of the user's hands a reported handedness label refers to.
        MediaPipe predicts handedness assuming a mirrored (selfie) image;
        depending on the camera pipeline the labels may arrive flipped.
        hand_labels_flipped toggles the mapping — verify with
        `preview --input hand`."""
        if self.cfg.hand_labels_flipped:
            return "Right" if label == "Left" else "Left"
        return label

    def read(self) -> tuple[np.ndarray | None, HandSample]:
        """Grab a frame and extract hand state. Frame is unmirrored BGR."""
        frame = self.camera.read()
        if frame is None:
            return None, HandSample()
        return frame, self.process(frame)

    def process(self, frame: np.ndarray) -> HandSample:
        """Run the landmarker on an externally captured BGR frame."""
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
            elif label == "Left":
                sample.left_ok = True
                sample.pinch_index = pinch_amount(lm, INDEX_TIP)
                sample.pinch_middle = pinch_amount(lm, MIDDLE_TIP)
        return sample

    def close(self):
        self.landmarker.close()
        if self._own_camera:
            self.camera.close()
