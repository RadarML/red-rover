"""Dataset loading utilities."""

import os
import json
import yaml
from functools import cached_property

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
        cfg: the original configuraton associated with collecting this dataset.
        sensors: dictionary of each non-virtual sensor in the dataset. The
            value is an initialized `SensorData` (or subclass).
    """

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

    def __getitem__(self, key: str) -> SensorData:
        """Alias for `self.sensors[...]`."""
        if key in self.sensors:
            return self.sensors[key]
        else:
            return SENSOR_TYPES.get(
                    self.cfg.get(key, {}).get("type", "u1"), SensorData
                )(os.path.join(self.path, key))

    def __repr__(self):
        """Get string representation."""
        return "{}({}: [{}])".format(
            self.__class__.__name__, self.path, ", ".join(self.cfg.keys()))
