"""IMU."""

import os
from functools import partial
from typing import Callable, overload

import numpy as np
from abstract_dataloader.generic import Metadata
from jaxtyping import Float64

from roverd import channels, timestamps, types

from .generic import Sensor


class IMU(Sensor[types.IMUData[np.ndarray], Metadata]):
    """IMU sensor.

    Args:
        path: path to sensor data directory. Must contain a `lidar.json` file
            with ouster lidar intrinsics.
        correction: optional timestamp correction to apply (i.e.,
            smoothing); can be a callable, string (name of a callable in
            [`roverd.timestamps`][roverd.timestamps]), or `None`. If `"auto"`,
            uses `smooth(interval=30.)`.
        past: number of past samples to include.
        future: number of future samples to include.
    """

    def __init__(
        self, path: str, correction: str | None | Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] = None,
        past: int = 0, future: int = 0
    ) -> None:
        if correction == "auto":
            correction = partial(timestamps.smooth, interval=30.)

        super().__init__(path, correction=correction, past=past, future=future)

        # Manual handling: on traces where we get a power cut, it's possible
        # that the entries are not the same length.
        ts = self.correction(self.channels["ts"].read(start=0, samples=-1))
        acc = self.channels["acc"].read(start=0, samples=-1)
        rot = self.channels["rot"].read(start=0, samples=-1)
        avel = self.channels["avel"].read(start=0, samples=-1)
        n = min(len(ts), len(acc), len(rot), len(avel))

        self.metadata = Metadata(ts[:n])
        self.imudata = types.IMUData(
            acc=acc[:n, None, ...], rot=rot[:n, None, ...],
            avel=avel[:n, None, ...], timestamps=ts[:n, None])

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
                acc=self.imudata.acc[
                    index - self.past:index + self.future + 1, 0][None],
                rot=self.imudata.rot[
                    index - self.past:index + self.future + 1, 0][None],
                avel=self.imudata.avel[
                    index - self.past:index + self.future + 1, 0][None],
                timestamps=self.imudata.timestamps[
                    index - self.past:index + self.future + 1, 0][None])


class Pose(Sensor[types.Pose[np.ndarray], Metadata]):
    """Pose sensor.

    Args:
        path: "virtual" path to the sensor. The actual pose data is read from
            a `.npz` file `_{reference}/pose.npz`.
        reference: reference sensor for the pose data.
        past: number of past samples to include.
        future: number of future samples to include.
    """

    def __init__(
        self, path: str, reference: str = "radar",
        past: int = 0, future: int = 0
    ) -> None:
        path = os.path.join(
            os.path.dirname(path), f"_{reference}", "pose.npz")
        self.pose = types.Pose(
            pos=np.load(path)["pos"].astype(np.float32)[:, None, ...],
            rot=np.load(path)["rot"].astype(np.float32)[:, None, ...],
            vel=np.load(path)["vel"].astype(np.float32)[:, None, ...],
            acc=np.load(path)["acc"].astype(np.float32)[:, None, ...],
            timestamps=np.load(path)["t"][:, None])
        self.metadata = Metadata(timestamps=self.pose.timestamps[:, 0])
        self.past = past
        self.future = future

    @overload
    def __getitem__(
        self, index: int | np.integer) -> types.Pose[np.ndarray]: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> types.Pose[np.ndarray] | channels.Channel:
        """Fetch pose data by index.

        Args:
            index: frame index, or channel name.

        Returns:
            Radar data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            raise ValueError(
                "Pose sensor does not support indexing by channel name.")
        else: # int | np.integer
            return types.Pose(
                pos=self.pose.pos[
                    index - self.past:index + self.future + 1, 0][None],
                rot=self.pose.rot[
                    index - self.past:index + self.future + 1, 0][None],
                vel=self.pose.vel[
                    index - self.past:index + self.future + 1, 0][None],
                acc=self.pose.acc[
                    index - self.past:index + self.future + 1, 0][None],
                timestamps=self.pose.timestamps[
                    index - self.past:index + self.future + 1, 0][None])
