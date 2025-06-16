"""Radar spectrum transforms.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from roverd.transforms import xwr
    ```

    You will also need to have the `xwr` extra installed.
"""

from typing import Literal, Type, cast

import numpy as np
from abstract_dataloader import spec
from xwr import rsp

from roverd import types


class Spectrum(spec.Transform[
    types.XWRRadarIQ[np.ndarray], types.XWR4DSpectrum[np.ndarray]
]):
    """Transform raw I/Q data to 4D spectrum data via FFT.

    Acts as a thin wrapper around [`xwr`](https://wiselabcmu.github.io/xwr/).

    Args:
        sensor: sensor type; see [`xwr.rsp`][xwr.rsp].
        window: whether to apply a hanning window. If `bool`, the same option
            is applied to all axes. If `dict`, specify per axis with keys
            "range", "doppler", "azimuth", and "elevation".
        size: target size for each axis after zero-padding, specified by axis.
            If an axis is not spacified, it is not padded.
    """

    def __init__(
        self, sensor: Type[rsp.BaseRSP] | str = "AWR1843", window: bool | dict[
            Literal["range", "doppler", "azimuth", "elevation"], bool] = False,
        size: dict[
            Literal["range", "doppler", "azimuth", "elevation"], int] = {}
    ) -> None:
        if isinstance(sensor, str):
            try:
                sensor = cast(Type[rsp.BaseRSP], getattr(rsp, sensor))
            except AttributeError:
                raise ValueError(f"Unknown sensor type: {sensor}.")

        self.rsp = sensor(window=window, size=size)

    def __call__(
        self, x: types.XWRRadarIQ[np.ndarray]
    ) -> types.XWR4DSpectrum[np.ndarray]:
        """Apply RSP."""
        return types.XWR4DSpectrum(
            spectrum=self.rsp(x.iq),
            timestamps=x.timestamps,
            range_resolution=x.range_resolution,
            doppler_resolution=x.doppler_resolution)
