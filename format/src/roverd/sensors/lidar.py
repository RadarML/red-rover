"""Lidar sensor."""

import os
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, overload

import numpy as np
from jaxtyping import Float64

from roverd import channels, timestamps, types

from .generic import Sensor


@dataclass
class LidarMetadata:
    """Lidar metadata.

    Attributes:
        timestamps: timestamp for each frame; nominally in seconds.
        intrinics: lidar intrinsics file; see the ouster sdk [`SensorInfo`](
            https://static.ouster.dev/sdk-docs/python/api/client.html#ouster.sdk.client.SensorInfo)
            documentation.
    """

    timestamps: Float64[np.ndarray, "N"]
    intrinsics: str


class OS0LidarDepth(Sensor[types.OS0Depth[np.ndarray], LidarMetadata]):
    """Ouster OS0 lidar sensor, depth/rng only.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        timestamp_interpolation: timestamp smoothing function to apply.
    """

    def __init__(self, path: str, timestamp_interpolation: Callable[
        [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] | None = None
    ) -> None:
        if timestamp_interpolation is None:
            timestamp_interpolation = partial(
                timestamps.discretize, interval=10., eps=0.05)

        super().__init__(path, timestamp_interpolation)

        if not os.path.exists(os.path.join(path, 'lidar.json')):
            warnings.warn(
                f"No 'lidar.json' found in {path}; using '' as a placeholder.")
            intrinsics = ""
        else:
            intrinsics = os.path.join(path, "lidar.json")

        self.metadata = LidarMetadata(
            timestamps=self.channels['ts'].read(start=0, samples=-1),
            intrinsics=intrinsics)

    @overload
    def __getitem__(self, index: int | np.integer) -> types.OS0Depth: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.OS0Depth[np.ndarray] | channels.Channel:
        """Read lidar data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else: # int | np.integer
            return types.OS0Depth(
                rng=self.channels['rng'][index],
                timestamps=self.metadata.timestamps[index][None],
                intrinsics=self.metadata.intrinsics)


class OS0Lidar(Sensor[types.OS0Data[np.ndarray], LidarMetadata]):
    """Ouster OS0 lidar sensor, all channels.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        timestamp_interpolation: timestamp smoothing function to apply.
    """

    def __init__(self, path: str, timestamp_interpolation: Callable[
        [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] | None = None
    ) -> None:
        if timestamp_interpolation is None:
            timestamp_interpolation = timestamps.smooth

        super().__init__(path, timestamp_interpolation)

        if not os.path.exists(os.path.join(path, 'lidar.json')):
            warnings.warn(
                f"No 'lidar.json' found in {path}; using '' as a placeholder.")
            intrinsics = ""
        else:
            intrinsics = os.path.join(path, "lidar.json")

        self.metadata = LidarMetadata(
            timestamps=self.channels['ts'].read(start=0, samples=-1),
            intrinsics=intrinsics)

    @overload
    def __getitem__(self, index: int | np.integer) -> types.OS0Data: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.OS0Data[np.ndarray] | channels.Channel:
        """Read lidar data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else: # int | np.integer
            return types.OS0Data(
                rng=self.channels['rng'][index],
                rfl=self.channels['rfl'][index],
                nir=self.channels['nir'][index],
                timestamps=self.metadata.timestamps[index][None],
                intrinsics=self.metadata.intrinsics)
