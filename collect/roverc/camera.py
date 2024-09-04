"""Camera data collection."""

import logging
import os
from queue import Queue

import cv2
import numpy as np
from beartype.typing import Optional, cast
from roverd import channels

from .common import Capture, Sensor, SensorException


class CameraCapture(Capture):
    """Camera capture data."""

    def __init__(
        self, path: str, fps: float = 1.0,
        report_interval: float = 5.0, log: Optional[logging.Logger] = None,
        width: int = 1920, height: int = 1080
    ) -> None:
        super().__init__(
            path=path, fps=fps, report_interval=report_interval, log=log)

        self.video: Queue = Queue()
        cast(channels.VideoChannel, self.sensor.create("video.avi", {
            "format": "mjpg", "type": "u1", "shape": (height, width, 3),
            "desc": "Ordinary camera video"
        })).consume(self.video, thread=True, fps=fps)

    def queue_length(self) -> int:
        return self.video.qsize()

    def write(self, data: np.ndarray) -> None:
        """Write MJPG stream."""
        self.video.put(data)

    def close(self) -> None:
        """Close files and clean up."""
        self.video.put(None)
        super().close()


class Camera(Sensor):
    """Video camera.

    Args:
        idx: camera index in `/dev`, i.e. `/dev/video0`.
        width, height: frame size, in pixels.
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
        self.log.info("Initialized camera {}: {}x{} @ {}fps".format(
            idx, width, height, fps))

    def capture(self, path: str) -> None:
        """Create capture (while `active` is set)."""
        out = CameraCapture(
            os.path.join(path, self.name), log=self.log,
            width=self.width, height=self.height, fps=self.fps)
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

        out.close()

    def close(self):
        self.cap.release()
