"""Dataset loading utilities."""

import json
import os
from functools import cached_property

import yaml

from .sensors import SENSOR_TYPES, SensorData


class Dataset:
    """A dataset with multiple sensors.

    Create a `Dataset` for the trace path; then, use `Dataset[...]` to
    fetch the associated sensors, then channels::

        ds = Dataset(path)
        radar = ds['radar']
        iq = radar['iq']

    Args:
        path: file path; should be a directory.

    Attributes:
        DEFAULT_SCHEMA: default schema of expected sensors and channels.
        cfg: the original configuraton associated with collecting this dataset.
        sensors: dictionary of each non-virtual sensor in the dataset. The
            value is an initialized `SensorData` (or subclass).
    """

    DEFAULT_SCHEMA = {
        "lidar": ["ts", "rfl", "nir", "rng"],
        "radar": ["ts", "iq", "valid"],
        "camera": ["ts", "video.avi"],
        "imu": ["ts", "rot", "acc", "avel"]
    }

    @staticmethod
    def find(*paths: list[str], follow_symlinks: bool = False) -> list[str]:
        """Walk a directory (or list of directories) to find all datasets.

        - Datasets are defined by directories containing a `config.yaml` file.
        - This method does not follow symlinks.

        Args:
            paths: a (list) of filepaths.
            follow_symlinks: whether to follow symlinks. If you have a circular
                symlink, and this is `True`, this method will loop infinitely!
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

    def __init__(self, path: str) -> None:
        self.path = path

        with open(os.path.join(self.path, "config.yaml")) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.sensors = {
            k: SENSOR_TYPES.get(
                self.cfg.get(k, {}).get("type", ""), SensorData
            )(os.path.join(self.path, k))
            for k in self.cfg.keys()}

    @cached_property
    def filesize(self):
        """Total filesize, iin bytes."""
        return sum(s.filesize for _, s in self.sensors.items())

    @cached_property
    def datarate(self):
        """Total data rate, in bytes/sec."""
        return sum(s.datarate for _, s in self.sensors.items())

    def create(
        self, key: str, exist_ok: bool = False,
        allow_physical: bool = True
    ) -> SensorData:
        """Intialize new sensor with an empty `meta.json` file.

        Args:
            key: sensor name.
            exist_ok: if `exist_ok=True` and the sensor already exists, that
                sensor is simply returned instead (similar to `os.mkdir`).

        Returns:
            Created sensor (or fetched, if it already exists and `exist_ok`).
        """
        if not allow_physical and not key.startswith('_'):
            raise ValueError(
                "Sensors must start with '_' unless they contain original "
                "collected data. If this is the case, you can override "
                "this error by setting `allow_physical=True`.")

        if os.path.exists(os.path.join(self.path, key, "meta.json")):
            if exist_ok:
                return self[key]
            else:
                raise ValueError("Sensor already exists: {}".format(key))

        os.makedirs(os.path.join(self.path, key), exist_ok=True)
        with open(os.path.join(self.path, key, "meta.json"), 'w') as f:
            json.dump({}, f)
        return self[key]

    def virtual_copy(
        self, key: str, exist_ok: bool = False
    ) -> SensorData:
        """Create a virtual sensor corresponding to an existing sensor.

        The virtual sensor will have the same name as the specified `key`, with
        a prepended `_`; timestamp data is copied as well.
        """
        original = self.sensors[key]
        copy = self.create(key='_' + key, exist_ok=exist_ok)
        if "ts" not in copy.channels:
            ts = copy.create("ts", original.config["ts"])
            ts.write(original["ts"].read())
        return copy

    def __getitem__(self, key: str) -> SensorData:
        """Alias for `self.sensors[...]`."""
        if key in self.sensors:
            return self.sensors[key]
        else:
            return SENSOR_TYPES.get(
                    self.cfg.get(key, {}).get("type", "u1"), SensorData
                )(os.path.join(self.path, key))

    def __contains__(self, key: str) -> bool:
        """Test whether this dataset contains the given sensor."""
        return os.path.exists(os.path.join(self.path, key, "meta.json"))

    def __repr__(self):
        """Get string representation."""
        return "{}({}: [{}])".format(
            self.__class__.__name__, self.path, ", ".join(self.cfg.keys()))
