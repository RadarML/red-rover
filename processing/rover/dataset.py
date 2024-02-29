"""Dataset loading utilities."""

import os
import json
import struct
from functools import cached_property
import numpy as np
from jaxtyping import Float64

from ouster import client

from .channel import BaseChannel, CHANNEL_TYPES


def smooth_timestamps(
    x: Float64[np.ndarray, "n"] , interval: float = 60.
) -> Float64[np.ndarray, "n"]:
    """Apply piecewise linear smoothing to system timestamps.
    
    Parameters
    ----------
    x: input timestamp array.
    interval: piecewise linear interpolation interval, in seconds.

    Returns
    -------
    Smoothed timestamp array.
    """
    blocksize = int(interval * (len(x) / (x[-1] - x[0])))
    start = 0
    out = np.zeros(x.shape, x.dtype)
    while x.shape[0] > 0:
        end = min(blocksize, x.shape[0])
        out[start:start + end] = np.linspace(x[0], x[end - 1], end)
        x = x[end:]
        start += end

    return out


class SensorData:
    """A sensor with multiple channels.
    
    Parameters
    ----------
    path: file path; should be a directory containing a `meta.json` file.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        try:
            with open(os.path.join(path, "meta.json")) as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(
                "{}: no valid 'metadata.json' found.".format(str(e)))

        self.channels = {k: self.open(k) for k in self.config}

    @cached_property
    def duration(self):
        """Trace duration."""
        with open(os.path.join(self.path, 'ts'), 'rb') as f:
            start, = struct.unpack('d', f.read(8))
            f.seek(-8, os.SEEK_END)
            end, = struct.unpack('d', f.read(8))
        return end - start

    @cached_property
    def filesize(self):
        """Total filesize."""
        return sum(c.filesize for _, c in self.channels.items())

    @cached_property
    def datarate(self):
        """Total data rate."""
        return self.filesize / self.duration

    def open(self, channel: str) -> BaseChannel:
        """Open channel."""
        cfg = self.config[channel]
        return CHANNEL_TYPES.get(cfg["format"], BaseChannel)(
            os.path.join(self.path, channel),
            dtype=cfg["type"], shape=cfg["shape"])

    def timestamps(self, interval: float = 60.0) -> Float64[np.ndarray, "n"]:
        """Get smoothed timestamps."""
        return smooth_timestamps(self.open("ts").read(), interval)

    def __getitem__(self, key: str) -> BaseChannel:
        """Alias for open channel."""
        return self.open(key)

    def __repr__(self):
        """Get string representation."""
        return "{}({}: [{}])".format(
            self.__class__.__name__, self.path, ", ".join(self.channels))

    def __len__(self):
        """Get number of messages."""
        return os.stat(os.path.join(self.path, "ts")).st_size // 8


class LidarData(SensorData):
    """Ouster Lidar sensor."""

    def lidar_metadata(self):
        """Get sensor metadata."""
        with open(os.path.join(self.path, "lidar.json")) as f:
            return client.SensorInfo(f.read())

    def pointcloud_stream(self):
        """Get an iterator which returns point clouds."""
        lut = client.XYZLut(self.lidar_metadata())
        return self.open("rng").stream_prefetch(transform=lut)


SENSOR_TYPES = {
    "lidar": LidarData
}


class Dataset:
    """A dataset with multiple sensors.
    
    Parameters
    ----------
    path: file path; should be a directory.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        _sensors = os.listdir(path)
        self.sensors = {
            s: SENSOR_TYPES.get(s, SensorData)(os.path.join(self.path, s))
            for s in _sensors
            if os.path.exists(os.path.join(path, s, "meta.json"))}

    @cached_property
    def filesize(self):
        """Total filesize."""
        return sum(s.filesize for _, s in self.sensors.items())

    @cached_property
    def datarate(self):
        """Total data rate."""
        return sum(s.datarate for _, s in self.sensors.items())

    def __getitem__(self, key: str) -> SensorData:
        """Alias for `self.sensors[...]`."""
        return self.sensors[key]

    def __repr__(self):
        """Get string representation."""
        return "{}({}: [{}])".format(
            self.__class__.__name__, self.path, ", ".join(self.sensors))
