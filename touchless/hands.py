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
WRIST, THUMB_TIP, INDEX_DIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP = 0, 4, 7, 8, 9, 12
# The four finger-base knuckles: their centroid is the wrist-mode focus
# point. Rigid under finger articulation, near the hand's rotation center,
# and averaging four landmarks halves the noise of any single one.
PALM_MCPS = (5, 9, 13, 17)


@dataclass
class HandSample:
    pointer_ok: bool = False
    pointer: np.ndarray | None = None   # (2,) right index tip, normalized cam coords
    pointer_rel: np.ndarray | None = None  # (2,) (index tip - palm centroid) / hand
                                           # size: hand-position/depth-invariant
    ref: np.ndarray | None = None       # (2,) palm-MCP centroid, RAW normalized
                                        # cam coords (arm-gate classifier input;
                                        # EMA'd would lag arm-onset detection)
    ref_px: np.ndarray | None = None    # (2,) palm-centroid focus point, px (HUD)
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
                min_hand_detection_confidence=cfg.hand_min_detection_confidence,
                # Low presence threshold on purpose: every dip below it
                # re-runs palm detection and the landmarks re-solve to a
                # slightly different answer - a persistent step the
                # stillness gate then has to absorb. Fewer re-solves
                # beats faster hand-lost detection here.
                min_hand_presence_confidence=cfg.hand_min_presence_confidence,
                min_tracking_confidence=cfg.hand_min_tracking_confidence,
            )
        )
        self._t0 = time.monotonic()
        self._last_ts_ms = -1
        self._size_ema: float | None = None
        self._ref_ema: np.ndarray | None = None

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
                # Blend the DIP joint into the pointer: the two landmarks'
                # noise is partly independent, so the mix is quieter than
                # the tip alone while tracking the same articulation.
                b = self.cfg.hand_pointer_blend
                tip = (1.0 - b) * lm[INDEX_TIP] + b * lm[INDEX_DIP]
                sample.pointer = tip
                size = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST])) + 1e-9
                # Hand size jitters frame to frame and multiplies straight
                # into pointer_rel, so normalize by a smoothed size instead.
                self._size_ema = (size if self._size_ema is None
                                  else 0.8 * self._size_ema + 0.2 * size)
                # Focus point = palm-knuckle centroid: rigid while the
                # index finger articulates (only the finger moves relative
                # to it), and averaging four landmarks halves per-landmark
                # noise (the old forearm extrapolation AMPLIFIED it 1.7x).
                # EMA on top: the centroid only moves with whole-hand
                # motion, which wrist mode ignores by design, so the lag
                # is nearly free.
                ref = lm[list(PALM_MCPS)].mean(axis=0)
                sample.ref = ref
                a = self.cfg.hand_wrist_ref_ema
                self._ref_ema = (ref if self._ref_ema is None
                                 else (1.0 - a) * self._ref_ema + a * ref)
                sample.pointer_rel = (tip - self._ref_ema) / self._size_ema
                sample.ref_px = self._ref_ema * np.array([w, h])
            elif label == "Left":
                sample.left_ok = True
                sample.pinch_index = pinch_amount(lm, INDEX_TIP)
                sample.pinch_middle = pinch_amount(lm, MIDDLE_TIP)
        if not sample.pointer_ok:
            self._size_ema = None  # hand lost: next sighting may be at a new depth
            self._ref_ema = None
        return sample

    def close(self):
        self.landmarker.close()
        if self._own_camera:
            self.camera.close()
