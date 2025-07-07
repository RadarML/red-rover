"""Lidar sensor."""

import os
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterator, overload

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


class OSLidarDepth(Sensor[types.OSDepth[np.ndarray], LidarMetadata]):
    """Ouster lidar sensor, depth/rng only.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        correction: optional timestamp correction to apply (i.e.,
            smoothing); can be a callable, string (name of a callable in
            [`roverd.timestamps`][roverd.timestamps]), or `None`. If `"auto"`,
            uses `discretize(interval=10., eps=0.05)`.
    """

    def __init__(self, path: str, correction: str | None | Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] = None
    ) -> None:
        if correction == "auto":
            correction = partial(timestamps.discretize, interval=10., eps=0.05)

        super().__init__(path, correction=correction)

        if not os.path.exists(os.path.join(path, 'lidar.json')):
            warnings.warn(
                f"No 'lidar.json' found in {path}; using '' as a placeholder.")
            intrinsics = ""
        else:
            intrinsics = os.path.join(path, "lidar.json")

        self.metadata = LidarMetadata(
            timestamps=self.correction(
                self.channels['ts'].read(start=0, samples=-1)),
            intrinsics=intrinsics)

    @overload
    def __getitem__(self, index: int | np.integer) -> types.OSDepth: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.OSDepth[np.ndarray] | channels.Channel:
        """Read lidar data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else: # int | np.integer
            return types.OSDepth(
                rng=self.channels['rng'][index][None],
                timestamps=self.metadata.timestamps[index][None, None],
                intrinsics=self.metadata.intrinsics)

    def stream(  # type: ignore
        self, batch: int | None = None
    ) -> Iterator[types.OSDepth[np.ndarray]]:
        """Stream lidar data.

        Args:
            batch: if specified, stream in batches of this size; otherwise,
                stream one frame at a time.

        Yields:
            Lidar data.
        """
        if batch is not None:
            raise NotImplementedError()

        for t, rng in zip(
            self.metadata.timestamps, self.channels['rng'].stream()
        ):
            yield types.OSDepth(
                rng=rng[None, None],
                timestamps=t[None, None],
                intrinsics=self.metadata.intrinsics)


class OSLidar(Sensor[types.OSData[np.ndarray], LidarMetadata]):
    """Ouster lidar sensor, all channels.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        correction: optional timestamp correction to apply (i.e.,
            smoothing); can be a callable, string (name of a callable in
            [`roverd.timestamps`][roverd.timestamps]), or `None`. If `"auto"`,
            uses `discretize(interval=10., eps=0.05)`.
    """

    def __init__(self, path: str, correction: str | None | Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] = None
    ) -> None:
        if correction == "auto":
            correction = partial(timestamps.discretize, interval=10., eps=0.05)

        super().__init__(path, correction=correction)

        if not os.path.exists(os.path.join(path, 'lidar.json')):
            warnings.warn(
                f"No 'lidar.json' found in {path}; using '' as a placeholder.")
            intrinsics = ""
        else:
            intrinsics = os.path.join(path, "lidar.json")

        self.metadata = LidarMetadata(
            timestamps=self.correction(
                self.channels['ts'].read(start=0, samples=-1)),
            intrinsics=intrinsics)

    @overload
    def __getitem__(self, index: int | np.integer) -> types.OSData: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.OSData[np.ndarray] | channels.Channel:
        """Read lidar data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else: # int | np.integer
            return types.OSData(
                rng=self.channels['rng'][index][None],
                rfl=self.channels['rfl'][index][None],
                nir=self.channels['nir'][index][None],
                timestamps=self.metadata.timestamps[index][None, None],
                intrinsics=self.metadata.intrinsics)
