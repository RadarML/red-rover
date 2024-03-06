"""Data loading utilities."""

import os
import lzma
import numpy as np
import cv2

from queue import Queue, Empty
from threading import Thread

from jaxtyping import Shaped
from beartype.typing import Union, cast, Iterator


DATA_TYPES = {
    "f64": np.float64, "f32": np.float32, "i16": np.int16,
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
            while True:
                try:
                    return self.queue.get(timeout=1.0)
                except Empty:
                    if self.done:
                        raise StopIteration


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

    def stream(self, transform=None, batch: int = 0) -> Iterator[np.ndarray]:
        """Get iterable data stream."""
        raise NotImplementedError()

    def stream_prefetch(
        self, transform=None, size: int = 64, batch: int = 0
    ) -> Iterator[np.ndarray]:
        """Stream with multi-threaded prefetching."""
        return Prefetch(
            self.stream(transform=transform, batch=batch), size=size)

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

    def stream(self, transform=None, batch: int = 0) -> Iterator[np.ndarray]:
        """Get iterable data stream.
        
        Parameters
        ----------
        transform: callable to apply to the read data.
        batch: batch size to read. If 0, load only a single sample and do not
            append an empty axis.
        """
        shape = self.shape if batch == 0 else [batch, *self.shape]
        size = self.size if batch == 0 else batch * self.size

        if transform is None:
            transform = lambda x: x

        with self._READ(self.path, 'rb') as fp:  # type: ignore
            while True:
                data = cast(bytes, fp.read(size))  # type: ignore
                if len(data) < size:
                    fp.close()
                    partial_batch = len(data) // self.size
                    if partial_batch > 0:
                        yield transform(
                            np.frombuffer(
                                data[:partial_batch * size], dtype=self.type
                            ).reshape([partial_batch, *self.shape]))
                    return
                yield transform(
                    np.frombuffer(data, dtype=self.type).reshape(shape))


class LzmaChannel(RawChannel):
    """LZMA-compressed binary data."""

    _READ = staticmethod(lzma.open)  # type: ignore


class VideoChannel(BaseChannel):
    """Video data."""

    def read(self) -> Shaped[np.ndarray, "..."]:
        """Read all data."""
        cap = cv2.VideoCapture(self.path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        return np.stack(frames)

    def stream(self, transform=None, batch: int = 0) -> Iterator[np.ndarray]:
        """Get iterable data stream."""
        if batch != 0:
            raise NotImplementedError("Batch loading not yet implemented.")

        if transform is None:
            transform = lambda x: x

        cap = cv2.VideoCapture(self.path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
        cap.release()
        return


CHANNEL_TYPES = {
    "raw": RawChannel,
    "lzma": LzmaChannel,
    "mjpg": VideoChannel
}
