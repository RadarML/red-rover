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
        correction: optional timestamp correction to apply (i.e.,
            smoothing); can be a callable, string (name of a callable in
            [`roverd.timestamps`][roverd.timestamps]), or `None`. If `"auto"`,
            uses `smooth(interval=30.)`.
        past: number of past samples to include.
        future: number of future samples to include.
    """

    def __init__(
        self, path: str, key: str = "video.avi",
        correction: str | None | Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] = None,
        past: int = 0, future: int = 0
    ) -> None:
        if correction == "auto":
            correction = partial(timestamps.smooth, interval=30.)

        super().__init__(path, correction=correction, past=past, future=future)
        self.metadata = generic.Metadata(
            timestamps=self.correction(
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
                image=self.channels[self.key].read(
                    index - self.past, samples=self.window)[None],
                timestamps=self.metadata.timestamps[
                    index - self.past:index + self.future + 1][None])


class Semseg(Sensor[types.CameraSemseg[np.ndarray], generic.Metadata]):
    """Generic camera semseg.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        key: semseg channel name.
        correction: optional timestamp correction to apply (i.e.,
            smoothing); can be a callable, string (name of a callable in
            [`roverd.timestamps`][roverd.timestamps]), or `None`. If `"auto"`,
            uses `smooth(interval=30.)`.
        past: number of past samples to include.
        future: number of future samples to include.
    """

    def __init__(
        self, path: str, key: str = "segment",
        correction: str | None | Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] = None,
        past: int = 0, future: int = 0
    ) -> None:
        if correction == "auto":
            correction = partial(timestamps.smooth, interval=30.)

        super().__init__(path, correction=correction, past=past, future=future)
        self.metadata = generic.Metadata(
            timestamps=self.correction(
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
                semseg=self.channels[self.key].read(
                    index - self.past, samples=self.window)[None],
                timestamps=self.metadata.timestamps[
                    index - self.past: index + self.future + 1][None])
