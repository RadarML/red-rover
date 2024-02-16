"""Camera data collection.
"""

import os
import cv2
import time
import sys
import subprocess
import struct
import json

from stats import DutyCycle


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

        # self._reset_odd_state()
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.meta = {
            "video.avi": {
                "format": "mjpg", "type": "u8", "shape": (height, width, 3),
                "description": "Ordinary camera video"},
            "ts": {
                "format": "raw", "type": "f64", "shape": (),
                "description": "System epoch time"}
        }

    def _reset_odd_state(self) -> None:
        """Reset "odd" state.
        
        Our capture card seems to fail every other time on linux. This seems
        to happen to all programs, including ffmpeg and v4l2.

        Workaround:
        - Try to read it.
        - If this fails, we are now in the "good" state.
        - If this succeeds, we are now in the "bad" state, so run it again.
        """
        cmd = [
            sys.executable, "-c",
            "import cv2; cv2.VideoCapture({}).read()".format(self.idx)]
        try:
            subprocess.run(cmd, timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        else:
            try:
                subprocess.run(cmd, timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

    def capture(self, path: str) -> None:

        os.makedirs(path)

        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(self.meta, f, indent=4)

        stats = DutyCycle()

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(
            os.path.join(path, "video.avi"), fourcc,
            self.fps, (self.width, self.height))
        ts = open(os.path.join(path, "timestamp.f64"), 'wb')
        for i in range(600):
            ret = self.cap.grab()
            t = time.time()
            ts.write(struct.pack('d', t))
            if ret:
                ret, frame = self.cap.retrieve()
                out.write(frame)
            else:
                print("Timed out: i={}".format(i))
                break

            stats.observe((time.time() - t) * self.fps)

        stats.print()
        out.release()
        ts.close()

    def close(self):
        self.cap.release()


cam = Camera(0)
cam.capture("test")
cam.close()
