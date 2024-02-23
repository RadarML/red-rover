
import os
import json
import lzma
import numpy as np
from jaxtyping import Shaped


DATA_TYPES = {
    "f64": np.float64, "f32": np.float32,
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32
}


class BaseChannel:

    def __init__(self, path: str, dtype: str, shape: list[int]) -> None:
        self.path = path
        self.type = DATA_TYPES[dtype]
        self.shape = shape
        self.size = np.prod(shape)

    def read(self) -> Shaped[np.ndarray, "..."]:
        """Read all data."""
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    def stream(self):
        return iter(self)


class RawChannel(BaseChannel):

    _READ = open

    def read(self) -> Shaped[np.ndarray, "..."]:
        """Read all data."""
        with self._READ(self.path, 'rb') as f:
            data = f.read()
        return np.frombuffer(data, dtype=self.type).reshape(-1, *self.shape)

    def __iter__(self):
        self.fp = self._READ(self.path, 'rb')
        return self
    
    def __next__(self):
        data = self.fp.read(self.size * np.dtype(self.type).itemsize)
        if len(data) == 0:
            self.fp.close()
            raise StopIteration
        return np.frombuffer(data, dtype=self.type).reshape(self.shape)


class LzmaChannel(RawChannel):

    _READ = lzma.open


CHANNEL_TYPES = {
    "raw": RawChannel,
    "lzma": LzmaChannel
}


class Sensor:
    def __init__(self, path: str) -> None:
        self.path = path
        try:
            with open(os.path.join(path, "meta.json")) as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(
                "{}: no valid 'metadata.json' found.".format(str(e)))

    def open(self, channel: str) -> BaseChannel:
        cfg = self.config[channel]
        try:
            return CHANNEL_TYPES[cfg["format"]](
                os.path.join(self.path, channel),
                dtype=cfg["type"], shape=cfg["shape"])
        except KeyError:
            raise ValueError("Unknown format: {}".format(cfg["format"]))



class Dataset:
    def __init__(self, path: str) -> None:
        self.path = path
        self.sensors = os.listdir(path)    

    def open(self, sensor: str) -> Sensor:
        return Sensor(os.path.join(self.path, sensor))
