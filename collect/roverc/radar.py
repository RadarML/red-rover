"""Radar data collection."""

import json
import logging
import os
from queue import Queue

import numpy as np
from xwr import XWRSystem, capture

from .common import Capture, Sensor


class RadarCapture(Capture):
    """Radar capture data.

    Args:
        path: directory path to write data to.
        fps: target framerate
        report_interval: interval for reporting sensor statistics, in seconds
        log: parent logger to use
        shape: radar shape.
    """

    def __init__(
        self, path: str, fps: float = 1.0, report_interval: float = 5.0,
        log: logging.Logger | None = None, shape: list[int] = []
    ) -> None:
        super().__init__(
            path=path, fps=fps, report_interval=report_interval, log=log)

        self.iq: Queue = Queue()
        self.sensor.create("iq", {
            "format": "raw", "type": "i2", "shape": shape,
            "desc": "Raw I/Q stream."}
        ).consume(self.iq, thread=True)

        self.valid: Queue = Queue()
        self.sensor.create("valid", {
            "format": "raw", "type": "u1", "shape": [],
            "desc": "True if this frame is complete (no zero-fill)."}
        ).consume(self.valid, thread=True)

    def queue_length(self) -> int:
        return self.iq.qsize()

    def write(self, data: capture.types.RadarFrame) -> None:
        """Write a single frame."""
        self.iq.put(data.data)
        self.valid.put(
            np.array(True, dtype=np.uint8) if data.complete
            else np.array(False, dtype=np.uint8))

    def close(self) -> None:
        """Close files and clean up."""
        self.iq.put(None)
        self.valid.put(None)
        super().close()


class Radar(Sensor):
    """TI AWR1843Boost Radar Sensor & DCA1000EVM capture card.

    See [`XWRSystem`][xwr.] for arguments.
    """

    def __init__(
        self, name: str = "radar", radar: dict = {}, capture: dict = {}
    ) -> None:
        super().__init__(name=name)
        self.radar = XWRSystem(radar=radar, capture=capture)

    def capture(self, path: str) -> None:
        """Create capture (while `active` is set)."""
        out = RadarCapture(
            os.path.join(path, self.name), log=self.log,
            shape=self.radar.config.raw_shape, fps=self.radar.fps)

        with open(os.path.join(path, self.name, "radar.json"), 'w') as f:
            json.dump(self.radar.config.as_intrinsics(), f, indent=4)

        stream = self.radar.stream()
        for scan in stream:
            out.start(scan.timestamp)
            out.write(scan)
            out.end()

            if not self.active:
                break

        self.radar.stop()
        out.close()
