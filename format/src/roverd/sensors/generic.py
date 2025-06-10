"""Sensor base class."""

import json
import os
import warnings
from functools import cached_property
from typing import Callable, Sequence, TypeVar, cast, overload

import numpy as np
from abstract_dataloader import abstract, generic
from jaxtyping import Float64

from roverd import channels

TSample = TypeVar("TSample")
TGenericSample = TypeVar("TGenericSample", bound=dict[str, np.ndarray])
TMetadata = TypeVar("TMetadata", bound=abstract.Metadata)


class Sensor(abstract.Sensor[TSample, TMetadata]):
    """Base sensor class, providing various utility methods.

    Args:
        path: path to sensor data directory. Must contain a `meta.json` file;
            see the dataset format specifications.
    """

    def __init__(self, path: str) -> None:

        self.path = path

        try:
            with open(os.path.join(path, "meta.json")) as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(
                "{}: no valid 'metadata.json' found.".format(str(e)))

        self.channels = {
            name: channels.from_config(
                path=os.path.join(self.path, name), **cfg)
            for name, cfg in self.config.items()}

    @cached_property
    def filesize(self):
        """Total filesize, in bytes."""
        return sum(c.filesize for _, c in self.channels.items())

    @cached_property
    def datarate(self):
        """Total data rate, in bytes/sec."""
        return self.filesize / self.duration

    def __len__(self) -> int:
        """Total number of measurements."""
        return os.stat(os.path.join(self.path, "ts")).st_size // 8

    def __repr__(self):
        return "{}({}: [{}])".format(
            self.__class__.__name__, self.path, ", ".join(self.channels))


class DynamicSensor(Sensor[TGenericSample, generic.Metadata]):
    """Generic sensor type with dynamically configured data types.

    Args:
        path: path to sensor data directory. Must contain a `meta.json` file;
            see the dataset format specifications.
        create: if `True`, create a new sensor at the specified path if one is
            not already present.
        exist_ok: if `True`, do not raise an error if `create=True` and the
            sensor already exists.
        subset: if specified, only read the listed channels.
        timestamp_interpolation: optional timestamp interpolation to apply;
            see [`roverd.timestamps`][roverd.timestamps].
    """

    def __init__(
        self, path: str, create: bool = False, exist_ok: bool = False,
        subset: Sequence[str] | None = None,
        timestamp_interpolation: Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] | None = None
    ) -> None:
        if create and not exist_ok:
            if os.path.exists(self.path):
                raise ValueError(
                    "`create=True`, but this sensor already exists!")

            os.makedirs(path)
            with open(os.path.join(path, "meta.json"), 'w') as f:
                json.dump({}, f)

        super().__init__(path=path)
        self.timestamp_interpolation = timestamp_interpolation
        self.subset = subset

    @cached_property
    def metadata(self) -> generic.Metadata:  # type: ignore
        if 'ts' not in self.channels:
            warnings.warn(
                f"Sensor metadata does not contain 'ts' channel: {self.path}.")
            return generic.Metadata(timestamps=np.array([], dtype=np.float64))

        ts = self.channels['ts'].read(start=0, samples=-1)
        if self.timestamp_interpolation is not None:
            ts = self.timestamp_interpolation(ts)
        return generic.Metadata(timestamps=ts)

    def _flush_config(self) -> None:
        """Flush configuration to disk.

        !!! danger

            This is an inherently dangerous operation; call with extreme
            caution!
        """
        with open(os.path.join(self.path, "meta.json"), 'w') as f:
            json.dump(self.config, f, indent=4)

    def create(self, channel: str, meta: dict) -> channels.Channel:
        """Create and open new channel.

        Args:
            channel: name of new channel.
            meta: metadata for the new channel.

        Returns:
            The newly created channel; sensor metadata is also flushed to disk,
                and the channel is registered with this `SensorData`.
        """
        self.config[channel] = meta
        self.channels[channel] = channels.from_config(
            path=os.path.join(self.path, channel), **meta)
        self._flush_config()
        return self.channels[channel]

    @overload
    def __getitem__(self, index: int | np.integer) -> TGenericSample: ...

    @overload
    def __getitem__(self, index: str) -> channels.Channel: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> TGenericSample | channels.Channel:
        """Fetch measurement from this sensor, by index.

        Args:
            index: measurement index, or channel name.

        Returns:
            Measurement data, or channel object if `index` is a string.
        """
        if isinstance(index, str):
            return self.channels[index]
        else:  # int | np.integer
            if self.subset is not None:
                data = {k: self.channels[k][index] for k in self.subset}
            else:
                data = {k: v[index] for k, v in self.channels.items()}
            return cast(TGenericSample, data)
