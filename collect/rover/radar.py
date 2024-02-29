"""Radar data collection."""

import os

from .common import BaseCapture, BaseSensor, SensorMetadata
from .radar_api import AWRSystem, dca_types, RadarConfig, CaptureConfig


class RadarCapture(BaseCapture):
    """Radar capture data."""

    def _init(
        self, path, shape: list[int] = [], **_
    ) -> SensorMetadata:        
        self.iq = open(os.path.join(path, "iq"), mode='wb')
        self.valid = open(os.path.join(path, "valid"), mode='wb')
        return {
            "iq": {
                "format": "raw", "type": "u16", "shape": shape,
                "desc": "Raw I/Q stream."},
            "valid": {
                "format": "raw", "type": "u8", "shape": [],
                "desc": "True if this frame is complete (no zero-fill)."}}

    def write(self, scan: dca_types.RadarFrame) -> None:
        """Write a single frame."""
        self.iq.write(scan.data.tobytes())
        self.valid.write(b'\x01' if scan.complete else b'\x00')

    def close(self) -> None:
        """Close files and clean up."""
        super().close()
        self.iq.close()


class Radar(BaseSensor):
    """TI AWR1843Boost Radar Sensor & DCA1000EVM capture card.
    
    See `AWRSystem` for arguments.
    """

    def __init__(
        self, name: str = "radar", radar: dict = {}, capture: dict = {}
    ) -> None:
        super().__init__(name=name)
        self.radar = AWRSystem(
            radar=RadarConfig(**radar), capture=CaptureConfig(**capture))

    def capture(self, path: str) -> None:
        """Create capture (while `active` is set)."""
        out = RadarCapture(
            os.path.join(path, self.name), log=self.log,
            shape=self.radar.config.shape, fps=self.radar.fps)

        stream = self.radar.stream()
        for scan in stream:
            out.start(scan.timestamp)
            out.write(scan)
            out.end()

            if not self.active:
                break

        self.radar.stop()
        out.close()
