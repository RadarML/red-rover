"""Ouster Lidar transforms.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from roverd.transforms import ouster
    ```

    You will also need to have the `ouster-sdk` extra installed.
"""

import os

import numpy as np
from abstract_dataloader import spec
from einops import rearrange
from ouster.sdk import client

from roverd import types


class ConfigCache:
    """Ouster sensor intrinsics cache."""

    def __init__(self) -> None:
        self._by_path: dict[str, client.SensorInfo] = {}  # type: ignore
        self._by_cfg: dict[str, client.SensorInfo] = {}  # type: ignore

    def __get_sensorinfo(self, cfg: str) -> client.SensorInfo:  # type: ignore
        # ouster-sdk is a naughty, noisy library
        # it is in fact so noisy, that we have cut it off at the os level...
        stdout = os.dup(1)
        os.close(1)
        info = client.SensorInfo(cfg)  # type: ignore
        os.dup2(stdout, 1)
        os.close(stdout)
        return info

    def __getitem__(self, path: str) -> client.SensorInfo:  # type: ignore
        """Get sensor intrinsics by path.

        - First checks if we have a `SensorInfo` loaded already for the given
            path.
        - If not, we then check if we have a `SensorInfo` loaded with an
            identical configuration.
        - If neither is found, we read the file and create a new `SensorInfo`.

        Args:
            path: Path to the sensor intrinsics file.

        Returns:
            Ouster `SensorInfo`.
        """
        # We've never seen this path before. Need to inspect it.
        if path not in self._by_path:
            try:
                with open(path) as f:
                    cfg = f.read()
            except FileNotFoundError as e:
                raise ValueError(
                    f"Lidar sensor intrinsics {path} does not exist.") from e

            # Truly new -> load
            if cfg not in self._by_cfg:
                self._by_cfg[cfg] = self.__get_sensorinfo(cfg)
            # Make a link
            self._by_path[path] = self._by_cfg[cfg]

        return self._by_path[path]


class Destagger(spec.Transform[types.OSDepth, types.Depth]):
    """Destagger Ouster Lidar depth data.

    Args:
        config: if provided, use this configuration cache (to share with other
            transforms).
    """

    def __init__(self, config: ConfigCache | None = None) -> None:
        if config is None:
            config = ConfigCache()
        self._config = config

    def __call__(self, data: types.OSDepth) -> types.Depth:
        """Destagger Ouster Lidar depth data.

        Args:
            data: Ouster Lidar depth data with staggered measurements.

        Returns:
            Depth data with destaggered measurements.
        """
        batch, t, el, az = data.rng.shape

        batch_last = rearrange(data.rng, "b t el az -> el az (b t)")
        destaggered_hwb = client.destagger(  # type: ignore
            self._config[data.intrinsics], batch_last)
        destaggered = rearrange(
            destaggered_hwb, "el az (b t) -> b t el az", b=batch, t=t)

        return types.Depth(rng=destaggered, timestamps=data.timestamps)


class PointCloud(spec.Transform[types.OSDepth, types.PointCloud]):
    """Calculate point cloud from Ouster Lidar depth data.

    Args:
        min_range: minimum range to include in the point cloud; if `None`,
            all points are included.
    """

    def __init__(self, min_range: float | None = None) -> None:
        self.cache = ConfigCache()
        self.lut_cache = {}
        self.min_range = min_range

    def __call__(self, data: types.OSDepth) -> types.PointCloud:
        """Convert Ouster Lidar depth data to point cloud.

        Args:
            data: Ouster Lidar depth data with staggered measurements.

        Returns:
            Point cloud data.
        """
        meta = self.cache[data.intrinsics]
        mid = id(meta)

        if mid not in self.lut_cache:
            self.lut_cache[mid] = client.XYZLut(meta)  # type: ignore

        xyz = []
        for frame in data.rng:
            pc = self.lut_cache[mid](frame).astype(np.float32)

            if self.min_range is not None:
                valid = (np.linalg.norm(pc, axis=-1) >= self.min_range)
            else:
                valid = np.any(pc != 0, axis=-1)
            valid = valid & np.all(~np.isnan(pc), axis=-1)

            xyz.append(pc.reshape(-1, 3)[valid.reshape(-1)])

        return types.PointCloud(
            xyz=np.concatenate(xyz, axis=0),
            length=np.array([len(x) for x in xyz], dtype=np.int32),
            timestamps=data.timestamps)
