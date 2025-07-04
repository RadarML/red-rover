"""Sensor base class."""

import json
import os
import warnings
from functools import cached_property
from typing import Callable, Sequence, TypeVar, cast, overload

import numpy as np
from abstract_dataloader import abstract, generic
from jaxtyping import Float64

from roverd import channels, timestamps

TSample = TypeVar("TSample")
TGenericSample = TypeVar("TGenericSample", bound=dict[str, np.ndarray])
TMetadata = TypeVar("TMetadata", bound=abstract.Metadata)


class Sensor(abstract.Sensor[TSample, TMetadata]):
    """Base sensor class, providing various utility methods.

    Args:
        path: path to sensor data directory. Must contain a `meta.json` file;
            see the dataset format specifications.
        correction: optional timestamp correction to apply (i.e.,
            smoothing); can be a callable, string (name of a callable in
            [`roverd.timestamps`][roverd.timestamps]), or `None`.

    Attributes:
        path: path to sensor data directory.
        channels: dictionary of channels, keyed by channel name.
        correction: timestamp correction function to apply.
    """

    def __init__(
        self, path: str,
        correction: str | None | Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] = None
    ) -> None:

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

        if correction is None:
            self.correction = timestamps.identity
        elif isinstance(correction, str):
            correction = getattr(timestamps, correction, None)
            if correction is None:
                raise ValueError(
                    f"Unknown timestamp correction function: {correction}")
            self.correction = cast(
                Callable[[Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]],
                correction)
        else:
            self.correction = correction

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
        correction: optional timestamp correction to apply (i.e.,
            smoothing); can be a callable, string (name of a callable in
            [`roverd.timestamps`][roverd.timestamps]), or `None`.
    """

    def __init__(
        self, path: str, create: bool = False, exist_ok: bool = False,
        subset: Sequence[str] | None = None,
        correction: str | None | Callable[
            [Float64[np.ndarray, "N"]], Float64[np.ndarray, "N"]] = None
    ) -> None:
        if create:
            if os.path.exists(path):
                if not exist_ok:
                    raise ValueError(
                        "`create=True`, but this sensor already exists!")
            else:
                os.makedirs(path)
                with open(os.path.join(path, "meta.json"), 'w') as f:
                    json.dump({}, f)

        super().__init__(path=path, correction=correction)
        self.subset = subset
        self.path = path

    @cached_property
    def metadata(self) -> generic.Metadata:  # type: ignore
        if 'ts' not in self.channels:
            warnings.warn(
                f"Sensor metadata does not contain 'ts' channel: {self.path}.")
            return generic.Metadata(timestamps=np.array([], dtype=np.float64))

        ts = self.channels['ts'].read(start=0, samples=-1)
        corrected = self.correction(ts)
        return generic.Metadata(timestamps=corrected)

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
