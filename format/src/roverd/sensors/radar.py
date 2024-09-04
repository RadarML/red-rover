"""TI radar sensor."""

import numpy as np
from beartype.typing import Iterator
from jaxtyping import Complex64, Int16

from .base import SensorData


class RadarData(SensorData):
    """TI Radar sensor.

    Radar data are stored in a non-standard `IIQQ int16` format; see
    `collect.radar_api.dca_types.RadarFrame` for details. Radar data should
    be converted to `complex64` in order to be used::

        radar = Dataset(path)["radar]
        raw = radar["iq"].read(1)
        sample = radar.iiqq16_to_iq64(raw)
        # or
        sample = RadarData.iiqq16_to_iq64(raw)
        # or
        for sample in radar["iq"].iq_stream():
            ...

    Note that the `IIQQ int16` format uses only 32-bytes per sample, while
    `complex64` uses 64-bytes per sample, so should not be used for storage.
    """

    @staticmethod
    def iiqq16_to_iq64(
        iiqq: Int16[np.ndarray, "... iiqq"]
    ) -> Complex64[np.ndarray, "... iq"]:
        """Convert IIQQ int16 to float64 IQ."""
        iq = np.zeros(
            (*iiqq.shape[:-1], iiqq.shape[-1] // 2), dtype=np.complex64)
        iq[..., 0::2] = 1j * iiqq[..., 0::4] + iiqq[..., 2::4]
        iq[..., 1::2] = 1j * iiqq[..., 1::4] + iiqq[..., 3::4]

        return iq

    def iq_stream(
        self, batch: int = 64, prefetch: bool = False
    ) -> Iterator[Complex64[np.ndarray, "..."]]:
        """Get an iterator which returns a Complex64 stream of IQ frames.

        NOTE: TI, for some reason, streams data in IIQQ order instead of IQIQ.
        This special stream (instead of a generic `.stream()`) handles this.

        Args:
            batch: batch size.
            prefetch: whether to prefetch data from another thread.

        Returns:
            Possibly prefetched iterator yielding complex IQ frames.
        """
        if prefetch:
            return self.channels["iq"].stream_prefetch(
                batch=batch, transform=self.iiqq16_to_iq64)
        else:
            return self.channels["iq"].stream(
                batch=batch, transform=self.iiqq16_to_iq64)
