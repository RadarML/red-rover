"""Data loading utilities."""

import os
import lzma
import numpy as np

from queue import Queue
from threading import Thread

from jaxtyping import Shaped
from beartype.typing import Union, cast


DATA_TYPES = {
    "f64": np.float64, "f32": np.float32,
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32
}


class Prefetch:
    """Simple prefetch queue wrapper.
    
    Parameters
    ----------
    iterator: any python iterator.
    size: prefetch buffer size.
    """

    def __init__(self, iterator, size: int = 64) -> None:
        self.iterator = iterator
        self.queue: Queue = Queue(maxsize=size)
        self.done = False

        self.thread = Thread(target=self._prefetch)
        self.thread.daemon = True
        self.thread.start()
    
    def _prefetch(self):
        for item in self.iterator:
            self.queue.put(item)
        self.done = True

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.done and self.queue.empty():
            raise StopIteration
        else:
            return self.queue.get()


class BaseChannel:
    """Sensor data channel.
    
    Parameters
    ----------
    path: file path.
    dtype: data type, or string name of dtype (e.g. u8, f32).
    shape: data shape.
    """

    def __init__(
        self, path: str, dtype: Union[str, type], shape: list[int]
    ) -> None:
        self.path = path
        self.type = DATA_TYPES.get(dtype, dtype)  # type: ignore
        self.shape = shape
        self.size = np.prod(shape) * np.dtype(self.type).itemsize
        self.filesize = os.stat(self.path).st_size

    def read(self) -> Shaped[np.ndarray, "..."]:
        """Read all data."""
        raise NotImplementedError()

    def stream(self, transform=None):
        """Get iterable data stream."""
        raise NotImplementedError()

    def stream_prefetch(self, transform=None, size: int = 64):
        """Stream with multi-threaded prefetching."""
        return Prefetch(self.stream(transform=transform), size=size)

    def __repr__(self):
        """Get string representation."""
        return "{}({}: {} x {})".format(
            self.__class__.__name__, self.path, self.type.__name__, self.shape)


class RawChannel(BaseChannel):
    """Raw (uncompressed) data."""

    _READ = staticmethod(open)

    def read(self) -> Shaped[np.ndarray, "..."]:
        """Read all data."""
        with self._READ(self.path, 'rb') as f:  # type: ignore
            data = cast(bytes, f.read())
        return np.frombuffer(data, dtype=self.type).reshape(-1, *self.shape)

    def stream(self, transform=None):
        """Get iterable data stream.
        
        Parameters
        ----------
        transform: callable to apply to the read data.
        """
        with self._READ(self.path, 'rb') as fp:  # type: ignore
            while True:
                data = cast(bytes, fp.read(self.size))  # type: ignore
                if len(data) < self.size:
                    fp.close()
                    return
                out = np.frombuffer(data, dtype=self.type).reshape(self.shape)
                if transform is None:
                    yield out
                else:
                    yield transform(out)


class LzmaChannel(RawChannel):
    """LZMA-compressed binary data."""

    _READ = staticmethod(lzma.open)  # type: ignore


CHANNEL_TYPES = {
    "raw": RawChannel,
    "lzma": LzmaChannel
}
