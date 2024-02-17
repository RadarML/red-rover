"""Camera data collection.
"""

import os
import cv2
import numpy as np

from common import BaseCapture


class CameraCapture(BaseCapture):
    """Camera capture data."""

    def __init__(
        self, path: str, width: int = 1920, height: int = 1080,
        fps: float = 60.0
    ) -> None:
        _meta = {
            "video.avi": {
                "format": "mjpg", "type": "u8", "shape": (height, width, 3),
                "description": "Ordinary camera video"}
        }
        super().__init__(path, _meta, fps=fps)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video = cv2.VideoWriter(
            os.path.join(path, "video.avi"), fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        """Write MJPG stream."""
        self.video.write(frame)

    def close(self) -> None:
        """Close files and clean up."""
        super().close()
        self.video.release()


class Camera:
    """Video camera implementation.
    
    Parameters
    ----------
    idx: camera index in `/dev`, i.e. `/dev/video0`.
    """

    def __init__(
        self, idx: int = 0, width: int = 1920, height: int = 1080,
        fps: float = 60.
    ) -> None:
        self.idx = idx
        self.fps = fps
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def capture(self, path: str) -> None:

        out = CameraCapture(
            path, width=self.width, height=self.height, fps=self.fps)
        out.start()
        for i in range(60 * 30):
            ret = self.cap.grab()
            out.start()
            if not ret:
                print("Timed out: i={}".format(i))
                break

            _, frame = self.cap.retrieve()
            out.write(frame)
            out.end()
            
            if (i + 1) % 120 == 0:
                out.reset_stats()

        out.close()

    def close(self):
        self.cap.release()


# cam = Camera(0)
# cam.capture("test/camera")
# cam.close()
