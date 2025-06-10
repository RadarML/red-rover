"""Radar sensor."""

import json
import os
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, overload

import numpy as np
from jaxtyping import Float32, Float64

from roverd import channels, timestamps, types

from .generic import Sensor


@dataclass
class RadarMetadata:
    """Radar metadata.

    Attributes:
        range_resolution: range resolution for the modulation used; nominally
            in meters.
        doppler_resolution: doppler resolution; nominally in m/s.
        timestamps: timestamp for each frame; nominally in seconds.
    """

    range_resolution: Float32[np.ndarray, "1"]
    doppler_resolution: Float32[np.ndarray, "1"]
    timestamps: Float64[np.ndarray, "N"]


class XWRRadar(Sensor[types.XWRRadarIQ[np.ndarray], RadarMetadata]):
    """Full spectrum 4D radar sensor.

    Args:
        path: path to sensor data directory. Must contain a `radar.json`
            file with `range_resolution` and `doppler_resolution` keys.
        timestamp_interpolation: timestamp smoothing function to apply.
    """

    def __init__(
        self, path: str, timestamp_interpolation: Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] | None = None
    ) -> None:
        if timestamp_interpolation is None:
            timestamp_interpolation = partial(timestamps.smooth, interval=30.)

        super().__init__(path)

        try:
            with open(os.path.join(path, "radar.json")) as f:
                radar_cfg = json.load(f)
                dr = radar_cfg["range_resolution"]
                dd = radar_cfg["doppler_resolution"]
        except KeyError as e:
            raise KeyError(
                f"{os.path.join(path, 'radar.json')} is missing a required "
                f"key: {str(e)}") from e
        except FileNotFoundError:
            warnings.warn(
                "No `radar.json` found; setting `dr=0` and `dd=0`. "
                "This may cause problems for radar processing later!")
            dr, dd = 0.0, 0.0

        self.metadata = RadarMetadata(
            doppler_resolution=np.array([dd], dtype=np.float32),
            range_resolution=np.array([dr], dtype=np.float32),
            timestamps=timestamp_interpolation(
                self.channels['ts'].read(start=0, samples=-1)))

    @overload
    def __getitem__(self, index: int | np.integer) -> types.XWRRadarIQ: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.XWRRadarIQ[np.ndarray] | channels.Channel:
        """Fetch IQ data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else: # int | np.integer
            return types.XWRRadarIQ(
                iq=self.channels['iq'][index],
                timestamps=self.metadata.timestamps[index][None],
                range_resolution=self.metadata.range_resolution,
                doppler_resolution=self.metadata.doppler_resolution,
                valid=self.channels['valid'][index])
