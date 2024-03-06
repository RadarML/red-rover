"""Cartographer output interface."""

import numpy as np
import pandas as pd

from beartype.typing import NamedTuple
from jaxtyping import Bool, Float64

from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import medfilt


class Poses(NamedTuple):
    """Discrete sampled poses."""

    t: Float64[np.ndarray, "N"]
    pos: Float64[np.ndarray, "N 3"]
    vel: Float64[np.ndarray, "N 3"]
    acc: Float64[np.ndarray, "N 3"]
    rot: Float64[np.ndarray, "N 3 3"]



class RawTrajectory(NamedTuple):
    """Raw trajectory from cartographer."""

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


class Trajectory:
    """Sensor trajectory.

    Notes
    -----
    The provided path should be the output of our cartographer processing
    pipeline, with the following columns:
    - `field.header.stamp` (in ns)
    - `field.transform.translation.{xyz}` (meters)
    - `field.transform.rotation.{xyzw}` (quaternion, in xyzw order)

    Parameters
    ----------
    path: Path to cartographer output `trajectory.csv` file.
    smoothing: Smoothing coefficient passed to `scipy.interpolate.splprep`; is
        divided by the number of samples `N`.
    start_threshold: Start threshold in meters; the trace is started when the
        sensor moves more than `start_threshold` from the starting position.
    filter_size: applies a median filter to the start distance to handle any
        initalization jitter.
    """

    def __init__(
        self, path: str, smoothing: float = 10.0,
        start_threshold: float = 1.0, filter_size: int = 5,
    ) -> None:
        raw = RawTrajectory.from_csv(path)
        origin_dist = medfilt(
            np.linalg.norm(raw.xyz - raw.xyz[0, None], axis=0), filter_size)
        start = np.argmax(origin_dist > start_threshold)
        end = np.argmax(origin_dist[::-1] > start_threshold)
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
        
        Parameters
        ----------
        t: input timestamps; can be in an arbitrary order.

        Returns
        -------
        poses: Poses; only valid timestamps are included.
        mask: Mask of valid timestamp elements.
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
