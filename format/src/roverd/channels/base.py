"""Channel base class."""

import os
from abc import ABC
from functools import cached_property
from queue import Queue, Empty
from threading import Thread

import numpy as np

from jaxtyping import Shaped
from beartype.typing import Union, Iterator, Iterable, Callable, Optional, Any


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


class Channel(ABC):
    """Sensor data channel.

    Args:
        path: file path.
        dtype: data type, or string name of dtype (e.g. `u1`, `f4`).
        shape: data shape.
    """

    def __init__(
        self, path: str, dtype: Union[str, type, np.dtype], shape: list[int]
    ) -> None:
        self.path = path
        self.type = np.dtype(dtype)
        self.shape = shape
        self.size = int(np.prod(shape) * np.dtype(self.type).itemsize)

    def buffer_to_array(self, data: bytes) -> Shaped[np.ndarray, "n ..."]:
        """Convert raw buffer to the appropriate type and shape."""
        return np.frombuffer(data, self.type).reshape(-1, *self.shape)

    @cached_property
    def filesize(self) -> int:
        """Get file size on disk in bytes."""
        return os.stat(self.path).st_size

    def read(
        self, start: int = 0, samples: int = -1
    ) -> Shaped[np.ndarray, "..."]:
        """Read data.

        Args:
            start: start index to read.
            samples: number of samples/frames to read. If `-1`, read all data.

        Returns:
            Read frames as an array, with a leading axis corresponding to
            the number of `samples`.
        """
        raise NotImplementedError(
            "`.read()` is not implemented for this channel type.")

    def _verify_type(self, data: Shaped[np.ndarray, "..."]) -> None:
        """Verify data shape and type."""
        if (
                len(self.shape) > 0 and
                tuple(data.shape[-len(self.shape):]) != tuple(self.shape)
            ):
            raise ValueError(f"Data shape {data.shape} does not match channel "
                f"shape {self.shape}.")
        if data.dtype != self.type:
            raise ValueError(f"Data type {data.dtype} does not match channel "
                f"type {self.type}.")

    def write(
        self, data: Shaped[np.ndarray, "..."], mode: str = 'wb'
    ) -> None:
        """Write data.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        raise NotImplementedError(
            "`.write()` is not implemented for this channel type.")

    def stream(
        self, transform: Optional[
            Callable[[Shaped[np.ndarray, "..."]], Any]
        ] = None, batch: int = 0
    ) -> Iterator[np.ndarray]:
        """Get iterable data stream.

        Args:
            transform: callable to apply to the read data. Should take a single
                sample or batch of samples, and can return an arbitrary type.
            batch: batch size to read. If 0, load only a single sample and do
                not append an empty axis.

        Returns:
            Iterator which yields successive frames.
        """
        raise NotImplementedError(
            "`.stream()` is not implemented for this channel type.")

    def consume(self, iterator: Iterable[np.ndarray]) -> None:
        """Consume iterable and write to file.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        raise NotImplementedError(
            "`.consume()` is not implemented for this channel type.")

    def stream_prefetch(
        self, transform: Optional[
            Callable[[Shaped[np.ndarray, "..."]], Any]
        ] = None, size: int = 64, batch: int = 0
    ) -> Iterator[np.ndarray]:
        """Stream with multi-threaded prefetching.

        Args:
            transform: callable to apply to the read data. Should take a single
                sample or batch of samples, and can return an arbitrary type.
            batch: batch size to read. If 0, load only a single sample and do
                not append an empty axis.
            size: prefetch buffer size.

        Returns:
            Iterator which yields successive frames, with core computations
            running in a separate thread.
        """
        return Prefetch(
            self.stream(transform=transform, batch=batch), size=size)

    def __repr__(self):
        """Get string representation."""
        return "{}({}: {} x {})".format(
            self.__class__.__name__, self.path, str(self.type), self.shape)
