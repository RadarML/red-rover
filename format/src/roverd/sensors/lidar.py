"""Ouster lidar."""

import os
from functools import cached_property

import numpy as np
from beartype.typing import Iterator
from jaxtyping import Float32, Float64, UInt16

from ._timestamps import discretize_timestamps
from .base import SensorData


class LidarData(SensorData):
    """Ouster Lidar sensor.

    Note that Lidar data are stored in a "staggered" format, and must be
    destaggered when used::

        lidar = Dataset(path)["lidar"]
        raw = lidar['rng'].read(1)
        sample = lidar.destagger(raw)

    To get a pointcloud, the ouster-supplied API is wrapped in `pointcloud`::

        points = lidar.pointcloud(raw)

    Use `*_stream` versions of `destagger` and `stream` to get an iterator
    of the transformed data::

        for depthmaps in lidar.destaggered_stream():
            ...

        for points in lidar.pointcloud_stream():
            ...
    """

    @cached_property
    def _ouster_client(self):
        from ouster.sdk import client
        return client

    @cached_property
    def lidar_metadata(self):
        """Get sensor metadata."""
        with open(os.path.join(self.path, "lidar.json")) as f:
            info_json = f.read()

        # ouster-sdk is a naughty, noisy library
        # it is in fact so noisy, that we have cut it off at the os level...
        stdout = os.dup(1)
        os.close(1)
        info = self._ouster_client.SensorInfo(info_json)  # type: ignore
        os.dup2(stdout, 1)
        os.close(stdout)

        return info

    @cached_property
    def xyzlut(self):
        """Point cloud LUT."""
        return self._ouster_client.XYZLut(self.lidar_metadata)  # type: ignore

    def pointcloud(self, arr) -> Float32[np.ndarray, "?n xyz"]:
        """Convert to pointcloud."""
        points = self.xyzlut(arr).astype(np.float32)
        valid = np.any(
            points != 0, axis=-1) | np.any(np.isnan(points), axis=-1)
        return points.reshape(-1, 3)[valid.reshape(-1)]

    def pointcloud_stream(
        self, prefetch: bool = True
    ) -> Iterator[Float32[np.ndarray, "?n xyz"]]:
        """Get an iterator which returns point clouds."""
        if prefetch:
            return self.channels["rng"].stream_prefetch(
                transform=self.pointcloud)
        else:
            return self.channels["rng"].stream(transform=self.pointcloud)

    def destagger(self, arr):
        """Destagger data."""
        return self._ouster_client.destagger(  # type: ignore
            self.lidar_metadata, arr)

    def destaggered_stream(
        self, key: str = 'rng'
    ) -> Iterator[UInt16[np.ndarray, "..."]]:
        """Get iterator which returns a destaggered range stream."""
        return self.channels[key].stream_prefetch(self.destagger)

    def timestamps(
        self, interval: float = 30.0, smooth: bool = True, **kwargs
    ) -> Float64[np.ndarray, "n"]:
        """Get smoothed timestamps."""
        if smooth:
            return discretize_timestamps(self.channels["ts"].read(), interval)
        else:
            return self.channels["ts"].read()
