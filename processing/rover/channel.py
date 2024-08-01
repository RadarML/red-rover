"""Data loading utilities."""

import os
import lzma
from functools import cached_property
from queue import Queue, Empty
from threading import Thread

import cv2
import numpy as np

from jaxtyping import Shaped, Array
from beartype.typing import Union, cast, Iterator, List


class Prefetch:
    """Simple prefetch queue wrapper.

    Args:
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

    Args:
        path: file path.
        dtype: data type, or string name of dtype (e.g. u8, f32).
        shape: data shape.
    """

    def __init__(
        self, path: str, dtype: Union[str, type, np.dtype], shape: list[int]
    ) -> None:
        self.path = path
        self.type = np.dtype(dtype)
        self.shape = shape
        self.size = int(np.prod(shape) * np.dtype(self.type).itemsize)

    @cached_property
    def filesize(self) -> int:
        """Get file size on disk."""
        return os.stat(self.path).st_size

    def read(self, samples: int = -1) -> Shaped[np.ndarray, "..."]:
        """Read all data."""
        raise NotImplementedError()

    def index(self, idx: int) -> Shaped[np.ndarray, "..."]:
        """Index into data."""
        raise NotImplementedError()

    def write(
        self, data: Shaped[Union[np.ndarray, Array], "..."], mode: str = 'wb'
    ) -> None:
        """Write data."""
        with open(self.path, mode) as f:
            f.write(data.tobytes())

    def stream(self, transform=None, batch: int = 0) -> Iterator[np.ndarray]:
        """Get iterable data stream."""
        raise NotImplementedError()

    def stream_prefetch(
        self, transform=None, size: int = 64, batch: int = 0
    ) -> Iterator[np.ndarray]:
        """Stream with multi-threaded prefetching."""
        return Prefetch(
            self.stream(transform=transform, batch=batch), size=size)

    def consume(self, iterator: Iterator[Union[np.ndarray, Array]]) -> None:
        """Consume iterator and write to file."""
        with open(self.path, 'wb') as f:
            for data in iterator:
                f.write(data.tobytes())

    def __repr__(self):
        """Get string representation."""
        return "{}({}: {} x {})".format(
            self.__class__.__name__, self.path, str(self.type), self.shape)


class RawChannel(BaseChannel):
    """Raw (uncompressed) data."""

    _READ = staticmethod(open)

    def read(self, samples: int = -1) -> Shaped[np.ndarray, "..."]:
        """Read all data."""
        with self._READ(self.path, 'rb') as f:  # type: ignore
            size = -1 if samples == -1 else samples * self.size
            data = cast(bytes, f.read(size))
        return np.frombuffer(data, dtype=self.type).reshape(-1, *self.shape)

    def stream(self, transform=None, batch: int = 0) -> Iterator[np.ndarray]:
        """Get iterable data stream.

        Args:
            transform: callable to apply to the read data.
            batch: batch size to read. If 0, load only a single sample and do
                not append an empty axis.
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

    def memmap(self) -> np.memmap:
        """Open memory mapped array."""
        return np.memmap(
            self.path, dtype=self.type, mode='r',
            shape=(self.filesize // self.size, *self.shape))

    def index(self, idx: int) -> Shaped[np.ndarray, "..."]:
        """Index into data."""
        return self.memmap()[idx]


class LzmaChannel(RawChannel):
    """LZMA-compressed binary data."""

    _READ = staticmethod(lzma.open)  # type: ignore

    def memmap(self) -> np.memmap:
        """Open memory mapped array."""
        raise Exception("Cannot mem-map a compressed channel.")


class VideoChannel(BaseChannel):
    """Video data."""

    def read(self, samples: int = -1) -> Shaped[np.ndarray, "..."]:
        """Read all data."""
        cap = cv2.VideoCapture(self.path)
        frames: List[np.ndarray] = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret and len(frames) + 1 != samples:
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
                yield transform(frame)
            else:
                break
        cap.release()
        return

    def index(self, idx: int) -> Shaped[np.ndarray, "..."]:
        """Index into data."""
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise Exception("Could not open video.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ret, frame = cap.read()
        if not ret:
            raise Exception("Could not read frame.")
        return frame


CHANNEL_TYPES = {
    "raw": RawChannel,
    "lzma": LzmaChannel,
    "mjpg": VideoChannel
}
