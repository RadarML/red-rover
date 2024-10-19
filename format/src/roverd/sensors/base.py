"""Sensor base class."""

import json
import os
import struct
from abc import ABC
from functools import cached_property

import numpy as np
from jaxtyping import Float64

from roverd.channels import CHANNEL_TYPES, Channel

from ._timestamps import smooth_timestamps


class SensorData(ABC):
    """A sensor with multiple channels.

    Args:
        path: file path; should be a directory containing a `meta.json` file.
        create: whether we are creating a new sensor.
        exist_ok: whether to raise an error (as a safety feature) if this
            channel already exists. Otherwise, if `create=True` and
            `exist_ok=True`, the existing channel is simply opened instead.

    Attributes:
        channels: dictionary of each channel in this sensor. Each value is an
            initialized :py:class:`Channel`.
    """

    def __init__(
        self, path: str, create: bool = False, exist_ok: bool = False
    ) -> None:
        self.path = path

        if not create or (os.path.exists(self.path) and exist_ok):
            try:
                with open(os.path.join(path, "meta.json")) as f:
                    self.config = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(
                    "{}: no valid 'metadata.json' found.".format(str(e)))
        else:
            if os.path.exists(self.path):
                raise ValueError(
                    "`create=True`, but this sensor already exists!")

            os.makedirs(path)
            with open(os.path.join(path, "meta.json"), 'w') as f:
                json.dump({}, f)
            self.config = {}

        self.channels = {
            name: CHANNEL_TYPES.get(cfg["format"], Channel)(
                os.path.join(self.path, name),
                dtype=cfg["type"], shape=cfg["shape"])
            for name, cfg in self.config.items()
        }

    @cached_property
    def duration(self):
        """Trace duration, in seconds."""
        with open(os.path.join(self.path, 'ts'), 'rb') as f:
            start, = struct.unpack('d', f.read(8))
            f.seek(-8, os.SEEK_END)
            end, = struct.unpack('d', f.read(8))
        return end - start

    @cached_property
    def filesize(self):
        """Total filesize, in bytes."""
        return sum(c.filesize for _, c in self.channels.items())

    @cached_property
    def datarate(self):
        """Total data rate, in bytes/sec."""
        return self.filesize / self.duration

    def _flush_config(self) -> None:
        """Flush configuration to disk.

        **NOTE**: this is an inherently dangerous operation -- call with
        extreme caution!
        """
        with open(os.path.join(self.path, "meta.json"), 'w') as f:
            json.dump(self.config, f, indent=4)

    def create(self, channel: str, meta: dict) -> Channel:
        """Create and open new channel.

        Args:
            channel: name of new channel.
            meta: metadata for the new channel.

        Returns:
            The newly created channel; sensor metadata is also flushed to disk,
            and the channel is registered with this `SensorData`.
        """
        self.config[channel] = meta
        self.channels[channel] = CHANNEL_TYPES.get(meta["format"], Channel)(
            os.path.join(self.path, channel),
            dtype=meta["type"], shape=meta["shape"])
        self._flush_config()
        return self.channels[channel]

    def timestamps(
        self, interval: float = 30.0, smooth: bool = True,
    ) -> Float64[np.ndarray, "n"]:
        """Get smoothed timestamps."""
        if smooth:
            return smooth_timestamps(self.channels["ts"].read(), interval)
        else:
            return self.channels["ts"].read()

    def __getitem__(self, key: str) -> Channel:
        """Alias for `SensorData.channels[key]`."""
        return self.channels[key]

    def __contains__(self, key: str) -> bool:
        """Check whether this sensor contains the given channel."""
        return key in self.channels

    def __repr__(self):
        """Get string representation."""
        return "{}({}: [{}])".format(
            self.__class__.__name__, self.path, ", ".join(self.channels))

    def __len__(self):
        """Get number of messages."""
        return os.stat(os.path.join(self.path, "ts")).st_size // 8
