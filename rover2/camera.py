"""Camera data collection."""

import os, sys
import cv2
import numpy as np

from common import BaseCapture, BaseSensor, SensorException


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


class Camera(BaseSensor):
    """Video camera.
    
    Parameters
    ----------
    idx: camera index in `/dev`, i.e. `/dev/video0`.
    width: frame width, in pixels.
    height: frame height, in pixels.
    fps: camera framerate. Make sure the camera/capture card supports this
        exact framerate!
    name: sensor name, i.e. "camera".
    """

    def __init__(
        self, idx: int = 0, width: int = 1920, height: int = 1080,
        fps: float = 60., name: str = "camera"
    ) -> None:
        super().__init__(name=name)

        self.idx = idx
        self.fps = fps
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.log.critical("Failed to open camera; is it connected?")
            raise SensorException

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def capture(self, path: str) -> None:
        """Create capture (while `active` is set)."""
        out = CameraCapture(
            path, width=self.width, height=self.height, fps=self.fps)
        out.start()
        i = 0
        while self.active:
            ret = self.cap.grab()
            out.start()
            if not ret:
                self.log.error("Capture timed out.")
                self.active = False
                break

            _, frame = self.cap.retrieve()
            out.write(frame)
            out.end()

            i = (i + 1) % 120
            if i == 0:
                out.reset_stats()

        out.close()

    def close(self):
        self.cap.release()


if __name__ == '__main__':
    try:
        Camera.from_config(*sys.argv[1:]).loop()
        exit(0)
    except SensorException:
        exit(-1)
