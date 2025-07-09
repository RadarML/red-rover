"""High level API for trace & dataset loading."""

import os
import traceback
import warnings
from functools import cached_property
from typing import Callable, Mapping, Sequence, TypeVar, cast, overload

import numpy as np
from abstract_dataloader import abstract, generic, spec

from .sensors import Sensor, from_config

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
        cls, path: str, sync: spec.Synchronization = generic.Empty(),
        sensors: Mapping[
            str, Callable[[str], Sensor] | str | None] | None = None,
        include_virtual: bool = False, name: str | None = None
    ) -> "Trace":
        """Create a trace from a directory containing a single recording.

        Sensor types can be specified by:

        - `None`: use the [`DynamicSensor`][roverd.sensors.DynamicSensor] type.
        - `"auto"`: return a known sensor type if applicable; see
            [`roverd.sensors`][roverd.sensors].
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

        initialized = {
            k: from_config(
                os.path.join(path, k), type=(k if v == "auto" else v))
            for k, v in sensors.items()
        }

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
        traces: traces which make up this dataset; must be `roverd` traces!
    """

    def __init__(self, traces: Sequence[Trace[TSample]]) -> None:
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
        cls, paths: Sequence[str],
        sync: spec.Synchronization = generic.Empty(),
        sensors: Mapping[
            str, Callable[[str], Sensor] | str | None] | None = None,
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
        traces = []
        for p in paths:
            traces.append(Trace.from_config(
                p, sync=sync, sensors=sensors,
                include_virtual=include_virtual))

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


@overload
def split(
    dataset: Trace[TSample],  start: float = 0.0, end: float = 0.0
) -> Trace[TSample]: ...

@overload
def split(
    dataset: Dataset[TSample],  start: float = 0.0, end: float = 0.0
) -> Dataset[TSample]: ...

def split(
    dataset: Dataset[TSample] | Trace[TSample],
    start: float = 0.0, end: float = 0.0
) -> Dataset[TSample] | Trace[TSample]:
    """Get sub-trace or sub-dataset.

    Args:
        dataset: trace or dataset to split.
        start: start of the split, as a proportion of the trace length (`0-1`).
        end: end of the split, as a proportion of the trace length (`0-1`).

    Returns:
        Trace or dataset with a contiguous subset of samples according to the
        start and end indices.
    """
    if not (0 <= start < end <= 1):
        raise ValueError(
            f"Invalid split range: {start} - {end} (must be in [0, 1])")

    if isinstance(dataset, Trace):
        # Make dummy indices if `None`.
        if dataset.indices is None:
            indices = {
                k: np.arange(len(v), dtype=np.int32)
                for k, v in dataset.sensors.items()}
        else:
            indices = dataset.indices

        for v in indices.values():
            istart = int(len(v) * start)
            iend = int(len(v) * end)
            break
        else:
            raise ValueError("There must be at least one sensor.")

        return Trace(
            sensors=dataset.sensors,
            sync={k: np.copy(v[istart:iend]) for k, v in indices.items()})

    else:  # Dataset
        return Dataset(traces=[
            split(t, start=start, end=end) for t in dataset.traces])
