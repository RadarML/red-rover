"""High level API for trace & dataset loading."""

import os
from functools import cached_property
from typing import Callable, Mapping, TypeVar, cast, overload

import numpy as np
from abstract_dataloader import abstract, spec

from roverd.sensors import DynamicSensor, Sensor

TSample = TypeVar("TSample")


class Trace(abstract.Trace[TSample]):
    """A single trace, containing multiple sensors.

    Type Parameters:
        - `Sample`: sample data type which this `Sensor` returns. As a
            convention, we suggest returning "batched" data by default, i.e.
            with a leading singleton axis.

    Args:
        sensors: sensors which make up this trace.
        sync: synchronization protocol used to create global samples from
            asynchronous time series. If `Mapping`; the provided indices are
            used directly; if `None`, sensors are expected to already be
            synchronous (equivalent to passing `{k: np.arange(N), ...}`).
        name: friendly name; should only be used for debugging and inspection.
    """

    @staticmethod
    def find_sensors(path: str, virtual: bool = False) -> list[str]:
        """Find all (non-virtual) sensors in a given directory."""
        def is_valid(p: str) -> bool:
            return (
                os.path.isdir(os.path.join(path, p))
                and (virtual or not p.startswith('_'))
                and os.path.exists(os.path.join(path, p, "meta.json")))

        return [p for p in os.listdir(path) if is_valid(p)]

    @classmethod
    def from_config(
        cls, path: str, sync: spec.Synchronization, sensors: Mapping[
            str, Sensor | Callable[[str], Sensor] | None] | None = None,
        include_virtual: bool = False, name: str | None = None
    ) -> "Trace":
        """Create a trace from a directory containing a single recording.

        Sensor types can be specified by:

        - `None`: use the [`DynamicSensor`][roverd.sensors.DynamicSensor] type.
        - `Callable[[str], Sensor]`: a sensor constructor, which has all
            non-path arguments closed on.
        - `Sensor`: an already initialized sensor instance.

        !!! info

            Sensors can also be inferred automatically (`sensors: None`), in
            which case we ind and load all sensors in the directory, excluding
            virtual sensors (those starting with `_`) unless
            `include_virtual=True`. Each sensor is then initialized as a
            `DynamicSensor`.

        Args:
            path: path to trace directory.
            sync: synchronization protocol.
            sensors: sensor types to use.
            include_virtual: if `True`, include virtual sensors as well.
            name: friendly name; if not provided, defaults to the given `path`.
        """
        if sensors is None:
            _sensors = Trace.find_sensors(path, virtual=include_virtual)
            sensors = {k: None for k in _sensors}

        initialized = {}
        for k, v in sensors.items():
            if isinstance(v, Sensor):
                initialized[k] = v
            elif v is None:
                initialized[k] = DynamicSensor(os.path.join(path, k))
            else:
                initialized[k] = v(os.path.join(path, k))

        # Ignore this type error here until abstract-dataloader switches to
        # `Mapping`.
        return cls(
            sensors=initialized, sync=sync,  # type: ignore
            name=path if name is None else name)

    @cached_property
    def filesize(self):
        """Total filesize, in bytes.

        !!! warning

            The trace must be initialized with all sensors for this
            calculation to be correct.
        """
        return sum(getattr(s, 'filesize', 0) for s in self.sensors.values())

    @cached_property
    def datarate(self):
        """Total data rate, in bytes/sec.

        !!! warning

            The trace must be initialized with all sensors for this
            calculation to be correct.
        """
        return sum(getattr(s, 'datarate', 0) for s in self.sensors.values())

    @overload
    def __getitem__(self, index: str) -> Sensor: ...

    @overload
    def __getitem__(self, index: int | np.integer) -> TSample: ...

    def __getitem__(
        self, index: int | np.integer | str
    ) -> TSample | spec.Sensor:
        """Get sample from sychronized index (or fetch a sensor by name).

        !!! tip

            For convenience, traces can be indexed by a `str` sensor name,
            returning that [`Sensor`][abstract_dataloader.spec.].

        Args:
            index: sample index, or sensor name.

        Returns:
            Loaded sample if `index` is an integer type, or the appropriate
                [`Sensor`][abstract_dataloader.spec.] if `index` is a `str`.
        """
        # We just want to overwrite the docstring.
        return super().__getitem__(index)

    def __len__(self) -> int:
        """Total number of sensor-tuple samples."""
        return super().__len__()


class Dataset(abstract.Dataset[TSample]):
    """A dataset, consisting of multiple traces.

    Type Parameters:
        - `Sample`: sample data type which this `Dataset` returns. As a
            convention, we suggest returning "batched" data by default, i.e.
            with a leading singleton axis.

    Args:
        traces: traces which make up this dataset.
    """

    def __init__(self, traces: list[spec.Trace[TSample]]) -> None:
        self.traces = traces

    @staticmethod
    def find_traces(
        *paths: str, follow_symlinks: bool = False
    ) -> list[str]:
        """Walk a directory (or list of directories) to find all datasets.

        Datasets are defined by directories containing a `config.yaml` file.

        !!! warning

            This method does not follow symlinks by default. If you have a
            cirular symlink, and `follow_symlinks=True`, this method will loop
            infinitely!

        Args:
            paths: a (list) of filepaths.
            follow_symlinks: whether to follow symlinks.
        """
        def _find(path) -> list[str]:
            if os.path.exists(os.path.join(path, "config.yaml")):
                return [path]
            else:
                contents = (
                    os.path.join(path, s.name) for s in os.scandir(path)
                    if s.is_dir(follow_symlinks=follow_symlinks))
                return sum((_find(c) for c in contents), start=[])

        return sum((_find(p) for p in paths), start=[])

    @classmethod
    def from_config(
        cls, paths: list[str], sync: spec.Synchronization, sensors: Mapping[
            str, Sensor | Callable[[str], Sensor] | None] | None = None,
        include_virtual: bool = False
    ) -> "Dataset":
        """Create a dataset from a list of directories containing recordings.

        Constructor arguments are forwarded to [`Trace.from_config`][^^.].

        Args:
            paths: paths to dataset directories.
            sync: synchronization protocol.
            sensors: sensor types to use.
            include_virtual: if `True`, include virtual sensors as well.
        """
        traces = [
            Trace.from_config(
                p, sync=sync, sensors=sensors, include_virtual=include_virtual)
            for p in paths]
        return cls(traces=cast(list[Trace[TSample]], traces))  # type: ignore

    def __getitem__(self, index: int | np.integer) -> TSample:
        """Fetch item from this dataset by global index.

        Args:
            index: sample index.

        Returns:
            loaded sample.

        Raises:
            IndexError: provided index is out of bounds.
        """
        # We just want to overwrite the docstring.
        return super().__getitem__(index)

    def __len__(self) -> int:
        """Total number of samples in this dataset."""
        return super().__len__()
