"""Dataset loading utilities."""

import os
import json
import yaml
import struct
from functools import cached_property
import numpy as np

from beartype.typing import Iterator
from jaxtyping import Float64, UInt16, Int16, Complex64, Float32

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

    def create(self, channel: str, meta: dict) -> BaseChannel:
        """Create and open new channel."""
        self.config[channel] = meta
        self.channels[channel] = self.open(channel)
        with open(os.path.join(self.path, "meta.json"), 'w') as f:
            json.dump(self.config, f, indent=4)
        return self.channels[channel]

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

    @cached_property
    def lidar_metadata(self):
        """Get sensor metadata."""
        with open(os.path.join(self.path, "lidar.json")) as f:
            return client.SensorInfo(f.read())

    @cached_property
    def xyzlut(self):
        """Point cloud LUT."""
        return client.XYZLut(self.lidar_metadata)

    def pointcloud(self, arr):
        """Convert to pointcloud."""
        return self.xyzlut(arr).astype(np.float32)

    def pointcloud_stream(self) -> Iterator[Float32[np.ndarray, "..."]]:
        """Get an iterator which returns point clouds."""
        return self.open("rng").stream_prefetch(transform=self.pointcloud)

    def destagger(self, arr):
        """Destagger data."""
        return client.destagger(self.lidar_metadata, arr)

    def destaggered_stream(
        self, key: str
    ) -> Iterator[UInt16[np.ndarray, "..."]]:
        """Get iterator which returns a destaggered range stream."""
        return self.open(key).stream_prefetch(self.destagger)


class RadarData(SensorData):
    """TI Radar sensor."""

    @staticmethod
    def iiqq16_to_iq64(
        iiqq: Int16[np.ndarray, "... iiqq"]
    ) -> Complex64[np.ndarray, "... iq"]:
        """Convert IIQQ int16 to float64 IQ.
        
        NOTE: See `radar_api.dca_types.RadarFrame for data format info."""
        iq = np.zeros(
            (*iiqq.shape[:-1], iiqq.shape[-1] // 2), dtype=np.complex64)
        iq[..., 0::2] = 1j * iiqq[..., 0::4] + iiqq[..., 2::4]
        iq[..., 1::2] = 1j * iiqq[..., 1::4] + iiqq[..., 3::4]

        return iq

    def iq_stream(
        self, batch: int = 64
    ) -> Iterator[Complex64[np.ndarray, "..."]]:
        """Get an iterator which returns a Complex64 stream of IQ frames.
        
        NOTE: TI, for some reason, streams data in IIQQ order instead of IQIQ.
        This special stream (instead of a generic `.stream()`) handles this.
        """
        return self.open("iq").stream(
            batch=batch, transform=self.iiqq16_to_iq64)


SENSOR_TYPES = {
    "lidar": LidarData,
    "_lidar": LidarData,
    "radar": RadarData
}


class Dataset:
    """A dataset with multiple sensors.
    
    Parameters
    ----------
    path: file path; should be a directory.

    Attributes
    ----------
    sensors: dictionary of each sensor in the dataset. The value is an
        initialized `SensorData` (or subclass).
    """

    def __init__(self, path: str) -> None:
        self.path = path

        with open(os.path.join(self.path, "config.yaml")) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        sensor_names = self.cfg.keys()

        self.sensors = {
            k: SENSOR_TYPES.get(
                self.cfg.get(k, {}).get("type", ""), SensorData
            )(os.path.join(self.path, k))
            for k in sensor_names}

    @cached_property
    def filesize(self):
        """Total filesize."""
        return sum(s.filesize for _, s in self.sensors.items())

    @cached_property
    def datarate(self):
        """Total data rate."""
        return sum(s.datarate for _, s in self.sensors.items())

    def get(self, key: str) -> SensorData:
        """Get a sensor, which may be a "virtual" sensor."""
        if key in self.sensors:
            return self.sensors[key]
        else:
            return SENSOR_TYPES.get(
                self.cfg.get(key, {}).get("type", ""), SensorData
            )(os.path.join(self.path, key))

    def create(self, key: str, exist_ok: bool = False) -> SensorData:
        """Intialize new sensor with an empty `meta.json` file."""
        assert key.startswith('_')
        if os.path.exists(os.path.join(self.path, key, "meta.json")):
            if exist_ok:
                return self.get(key)
            else:
                raise ValueError("Sensor already exists: {}".format(key))

        os.makedirs(os.path.join(self.path, key), exist_ok=True)
        with open(os.path.join(self.path, key, "meta.json"), 'w') as f:
            json.dump({}, f)
        return self.get(key)

    def __getitem__(self, key: str) -> SensorData:
        """Alias for `self.sensors[...]`."""
        return self.sensors[key]

    def __repr__(self):
        """Get string representation."""
        return "{}({}: [{}])".format(
            self.__class__.__name__, self.path, ", ".join(self.sensors))
