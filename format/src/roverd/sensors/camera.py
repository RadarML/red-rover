"""Camera sensor."""

from functools import partial
from typing import Callable, overload

import numpy as np
from abstract_dataloader import generic
from jaxtyping import Float64

from roverd import channels, timestamps, types

from .generic import Sensor


class Camera(Sensor[types.CameraData[np.ndarray], generic.Metadata]):
    """Generic RGB camera.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        key: video channel name.
        timestamp_interpolation: timestamp smoothing function to apply.
    """

    def __init__(
        self, path: str, key: str = "video.avi",
        timestamp_interpolation: Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] | None = None
    ) -> None:
        if timestamp_interpolation is None:
            timestamp_interpolation = partial(timestamps.smooth, interval=30.)

        super().__init__(path)
        self.metadata = generic.Metadata(
            timestamps=timestamp_interpolation(
                self.channels["ts"].read(start=0, samples=-1)))
        self.key = key

    @overload
    def __getitem__(self, index: int | np.integer) -> types.CameraData: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.CameraData[np.ndarray] | channels.Channel:
        """Read camera data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else: # int | np.integer
            return types.CameraData(
                image=self.channels[self.key][index],
                timestamps=self.metadata.timestamps[index][None])


class Semseg(Sensor[types.CameraSemseg[np.ndarray], generic.Metadata]):
    """Generic camera semseg.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        key: semseg channel name.
        timestamp_interpolation: timestamp smoothing function to apply.
    """

    def __init__(
        self, path: str, key: str = "segment",
        timestamp_interpolation: Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] | None = None
    ) -> None:
        if timestamp_interpolation is None:
            timestamp_interpolation = partial(timestamps.smooth, interval=30.)

        super().__init__(path)
        self.metadata = generic.Metadata(
            timestamps=timestamp_interpolation(
                self.channels["ts"].read(start=0, samples=-1)))
        self.key = key

    @overload
    def __getitem__(self, index: int | np.integer) -> types.CameraSemseg: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.CameraSemseg[np.ndarray] | channels.Channel:
        """Read camera data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else: # int | np.integer
            return types.CameraSemseg(
                semseg=self.channels[self.key][index],
                timestamps=self.metadata.timestamps[index][None])
