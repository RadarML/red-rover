"""Radar data collection."""

import lzma
import os
import numpy as np


from common import BaseCapture, BaseSensor, SensorMetadata


class RadarCapture(BaseCapture):
    """Radar capture data."""

    def _init(
        self, path, chirps: int = 1, tx: int = 3, rx: int = 4,
        samples: int = 256, **_
    ) -> SensorMetadata:        
        self.iq = lzma.open(os.path.join(path, "iq"), mode='wb', preset=1)
        return {"iq": {
            "format": "lzma", "type": "u16", "shape": (
                (chirps, tx, rx, samples, 2))}}

    def write(self, data: np.ndarray) -> None:
        """Write a single frame."""
        self.iq.write(data.tobytes())

    def close(self) -> None:
        """Close files and clean up."""
        super().close()
        self.iq.close()


class Radar(BaseSensor):
    """TI AWR1843Boost Radar Sensor & DCA1000EVM capture card.
    
    Parameters
    ----------
    chirps: number of chirps per frame. Use `chirps=1` for virtual frames.
    fps: number of frames per second.
    name: sensor name, i.e. "radar".
    """

    def __init__(
        self, chirps: int = 1,
        samples: int = 256, fps: float = 2000.0, name: str = "radar"
    ) -> None:
        super().__init__(name=name)
        self._params = {
            "chirps": chirps, "tx": 3, "rx": 4, "samples": samples, "fps": fps}
        self.fps = fps

        self.awr = None
        self.awr.setup()
        self.dca = None
        self.dca.setup()

    def capture(self, path: str) -> None:
        """Create capture (while `active` is set)."""
        out = RadarCapture(
            os.path.join(path, self.name), log=self.log, **self._params)

        self.awr.start()
        self.dca.start()
        stream = self.dca.stream()
        for scan in stream:
            out.start(scan.timestamp)
            out.write(scan.data)
            out.end()

            if not self.active:
                break

        self.awr.stop()
        self.dca.stop()
        out.close()


if __name__ == '__main__':
    Radar.main()
