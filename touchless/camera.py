"""Webcam capture, separated from the landmarkers so multiple trackers
(face + hands) can share one camera — a device can only be opened once."""

import cv2
import numpy as np

from .config import Config


class Camera:
    def __init__(self, cfg: Config):
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(cfg.camera_index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {cfg.camera_index}. "
                "Try a different --camera index, and check no other app is using it."
            )

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()
