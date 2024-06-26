"""Cartographer output interface."""

import numpy as np
import pandas as pd

from beartype.typing import NamedTuple
from jaxtyping import Bool, Float64, Float, Integer

from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import medfilt


class Poses(NamedTuple):
    """Discrete sampled poses.
    
    Attributes:
        t: timestamp, in seconds.
        pos: position, in meters (front-left-up coordinate convention).
        vel: velocity, in meters/second.
        acc: acceleration, in meters/second^2.
        rot: rotation (as matrix).
    """

    t: Float64[np.ndarray, "N"]
    pos: Float64[np.ndarray, "N 3"]
    vel: Float64[np.ndarray, "N 3"]
    acc: Float64[np.ndarray, "N 3"]
    rot: Float64[np.ndarray, "N 3 3"]


class RawTrajectory(NamedTuple):
    """Raw trajectory from cartographer.
    
    Attributes:
        xyz: position
        quat: rotation (as quaternion)
        t: timestamp
    """

    xyz: Float64[np.ndarray, "3 N"]
    quat: Float64[np.ndarray, "4 N"]
    t: Float64[np.ndarray, "N"]

    @classmethod
    def from_csv(cls, path: str) -> "RawTrajectory":
        """Load `trajectory.csv` output file."""
        df = pd.read_csv(path)
        return RawTrajectory(
            xyz=np.stack([
                df["field.transform.translation.{}".format(axis)]
                for axis in "xyz"]),
            quat=np.stack([
                df["field.transform.rotation.{}".format(axis)]
                for axis in "xyzw"]),
            t=np.array(df['field.header.stamp']) / 1e9)

    def bounds(
        self, margin_xy: float = 5.0, margin_z: float = 5.0,
        resolution: float = 50.0, align: int = 16
    ) -> tuple[
        Float[np.ndarray, "3"],
        Float[np.ndarray, "3"],
        Integer[np.ndarray, "3"]
    ]:
        """Get grid bounds.
        
        Args:
            margin_xy, margin_z: grid margin around trajectory bounds in the
                horizontal plane and vertical axis, respectively.
            resolution: resolution, in grid cells/m.
            align: grid resolution alignment; the grid will be expanded from the
                specified margin until the resolution is divisible by `align`.

        Returns:
            (lower bound, upper bound, grid size)        
        """
        margin = np.array([margin_xy, margin_xy, margin_z])
        lower = np.min(self.xyz, axis=1) - margin
        upper = np.max(self.xyz, axis=1) + margin

        size = ((upper - lower) * resolution + align - 1) // align * align

        round_up = (size / resolution) - (upper - lower)
        lower = lower - round_up / 2
        upper = upper + round_up / 2
        return lower, upper, size.astype(int)


class Trajectory:
    """Sensor trajectory.

    The provided path should be the output of our cartographer processing
    pipeline, with the following columns:

    - `field.header.stamp` (in ns)
    - `field.transform.translation.{xyz}` (meters)
    - `field.transform.rotation.{xyzw}` (quaternion, in xyzw order)

    Args:
        path: Path to cartographer output `trajectory.csv` file.
        smoothing: Smoothing coefficient passed to `scipy.interpolate.splprep`;
            is divided by the number of samples `N`.
        start_threshold: Start threshold in meters; the trace is started when
            the sensor moves more than `start_threshold` from the starting
            position.
        filter_size: applies a median filter to the start distance to handle
            any initalization jitter.
    """

    def __init__(
        self, path: str, smoothing: float = 10.0,
        start_threshold: float = 1.0, filter_size: int = 5,
    ) -> None:
        raw = RawTrajectory.from_csv(path)
        origin_dist = medfilt(
            np.linalg.norm(raw.xyz - raw.xyz[0, None], axis=0), filter_size)
        start = np.argmax(origin_dist > start_threshold)
        end = max(1, np.argmax(origin_dist[::-1] > start_threshold))

        _t_slam = raw.t[start:-end]

        self.xyz = raw.xyz[:, start:-end]
        self.quat = raw.quat[:, start:-end]
        self.base_time = _t_slam[0]
        self.t_slam = _t_slam - self.base_time

        self.tck, *_ = splprep(
            self.xyz, u=self.t_slam, s=smoothing / self.t_slam.shape[0])
        self.slerp = Slerp(self.t_slam, Rotation.from_quat(self.quat.T))

    def interpolate(
        self, t: Float64[np.ndarray, "N"]
    ) -> tuple[Poses, Bool[np.ndarray, "N"]]:
        """Interpolate trajectory to target timestamps.
        
        Args:
            t: input timestamps; can be in an arbitrary order.

        Returns:
            (poses, mask). Only valid timestamps are included in `poses`;
            these valid timestamps are specified in `mask`.
        """

        t_rel = t - self.base_time
        mask = (t_rel > 0) & (t_rel < self.t_slam[-1])
        t_valid = t_rel[mask]

        pos = np.array(splev(t_valid, self.tck)).T
        vel = np.array(splev(t_valid, self.tck, der=1)).T
        acc = np.array(splev(t_valid, self.tck, der=2)).T
        rot = self.slerp(t_valid).as_matrix()

        return Poses(
            t=t_valid + self.base_time,
            pos=pos, vel=vel, rot=rot, acc=acc), mask
