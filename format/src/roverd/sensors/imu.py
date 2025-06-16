"""IMU."""

from functools import partial
from typing import Callable, overload

import numpy as np
from jaxtyping import Float64

from roverd import channels, timestamps, types

from .generic import Sensor


class IMU(Sensor[types.IMUData[np.ndarray], types.IMUData[np.ndarray]]):
    """IMU sensor.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        correction: optional timestamp correction to apply (i.e.,
            smoothing); can be a callable, string (name of a callable in
            [`roverd.timestamps`][roverd.timestamps]), or `None`. If `"auto"`,
            uses `smooth(interval=30.)`.
    """

    def __init__(
        self, path: str, correction: str | None | Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] = None
    ) -> None:
        if correction == "auto":
            correction = partial(timestamps.smooth, interval=30.)

        super().__init__(path, correction=correction)

        # Manual handling: on traces where we get a power cut, it's possible
        # that the entries are not the same length.
        ts = self.correction(self.channels["ts"].read(start=0, samples=-1))
        acc = self.channels["acc"].read(start=0, samples=-1)
        rot = self.channels["rot"].read(start=0, samples=-1)
        avel = self.channels["avel"].read(start=0, samples=-1)
        n = min(len(ts), len(acc), len(rot), len(avel))

        self.metadata = types.IMUData(
            acc=acc[:n], rot=rot[:n], avel=avel[:n], timestamps=ts[:n])

    @overload
    def __getitem__(
        self, index: int | np.integer) -> types.IMUData[np.ndarray]: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.IMUData[np.ndarray] | channels.Channel:
        """Fetch IMU data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else: # int | np.integer
            return types.IMUData(
                acc=self.metadata.acc[index][None],
                rot=self.metadata.rot[index][None],
                avel=self.metadata.avel[index][None],
                timestamps=self.metadata.timestamps[index][None])
