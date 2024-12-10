"""Channel base class."""

import os
from abc import ABC
from functools import cached_property
from queue import Queue
from threading import Thread

import numpy as np
from beartype.typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TypeVar,
    cast,
)
from jaxtyping import Shaped

Data = np.ndarray | bytes | bytearray
"""Generic writable data.

Should generally behave as follows:
- If `Shaped[np.ndarray, "..."]`, the shape and dtype are assumed to have
  semantic meaning, and are verified.
- If `bytes` or `bytearray`, we assume that the caller has already done any
  necessary binary conversion. No type or shape verification is performed.
"""


T = TypeVar("T")

Streamable = Iterator[T] | Iterable[T] | Queue[T]
"""Any stream-like container."""


class Buffer(Generic[T]):
    """Simple queue buffer (i.e. queue to iterator) with batching.

    Args:
        queue: queue to use as a buffer. Should return `None` when the stream
            is complete (i.e. `StopIteration`).
    """

    def __init__(self, queue: Queue[T]) -> None:
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self) -> T:
        item = self.queue.get()
        if item is None:
            raise StopIteration
        else:
            return item


class Prefetch(Buffer):
    """Simple prefetch queue wrapper (i.e. iterator to queue).

    Can be used as a prefetched iterator (`for x in Prefetch(...)`), or as a
    queue (`Prefetch(...).queue`). When used as a queue, `None` is put in the
    queue to indicate that the iterator has terminated.

    Args:
        iterator: any python iterator; must never yield `None`.
        size: prefetch buffer size.
    """

    def __init__(
        self, iterator: Iterable[T] | Iterator[T], size: int = 64
    ) -> None:
        super().__init__(queue=Queue(maxsize=size))
        self.iterator = iterator

        Thread(target=self._prefetch, daemon=True).start()

    def _prefetch(self) -> None:
        for item in self.iterator:
            self.queue.put(item)
        self.queue.put(None)


def batch_iterator(
    iterator: Iterator[T] | Iterable[T], size: int = 8
) -> Iterator[list[T]]:
    """Convert an iterator into a batched version.

    Args:
        iterator: input iterator/iterable.
        size: batch size.
    """
    buf = []
    for item in iterator:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) != 0:
        yield buf


class Channel(ABC):
    """Sensor data channel.

    Args:
        path: file path.
        dtype: data type, or string name of dtype (e.g. `u1`, `f4`).
        shape: data shape.

    Attributes:
        path: file path.
        type: numpy data type.
        shape: sample data shape.
        size: total file size, in bytes.
    """

    def __init__(
        self, path: str, dtype: str | type | np.dtype, shape: list[int]
    ) -> None:
        self.path = path
        self.type = np.dtype(dtype)
        self.shape = shape
        self.size = int(np.prod(shape) * np.dtype(self.type).itemsize)

    def open_like(self, path: str) -> "Channel":
        """Open a channel with the same metadata, but different data."""
        return self.__class__(path=path, dtype=self.type, shape=self.shape)

    def buffer_to_array(
        self, data: bytes, batch: bool = True
    ) -> Shaped[np.ndarray, "n ..."]:
        """Convert raw buffer to the appropriate type and shape."""
        arr = np.frombuffer(data, self.type).reshape(-1, *self.shape)
        if batch:
            return arr
        else:
            if arr.shape[0] != 1:
                raise ValueError(
                    "`batch=False`, but received `data` which contains more "
                    "than one item.")
            return arr[0]

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

    def _verify_type(self, data: Data | Sequence[Data]) -> None:
        """Verify data shape and type.

        Implementation notes:
        - If `data` is batched as a `list | tuple`, only the first element is
          checked.
        - If `data` (or its contents) are not a `np.ndarray`, (i.e. is
          `bytes | bytearray`), we assume the caller has correctly serialized
           the data, and type checking always succeeds.
        """
        if isinstance(data, (list, tuple)):
            data = data[0]
        if not isinstance(data, np.ndarray):
            return

        if (
                len(self.shape) > 0 and
                tuple(data.shape[-len(self.shape):]) != tuple(self.shape)
            ):
            raise ValueError(f"Data shape {data.shape} does not match channel "
                f"shape {self.shape}.")
        if data.dtype != self.type:
            raise ValueError(f"Data type {data.dtype} does not match channel "
                f"type {self.type}.")

    def _serialize(
        self, data: Data | Sequence[Data]
    ) -> bytes | bytearray:
        """Serialize data into a binary format for writing."""
        if isinstance(data, (list, tuple)):
            return b''.join(self._serialize(x) for x in data)

        data = cast(Data, data)
        if isinstance(data, np.ndarray):
            return data.data
        else:
            return data

    def write(
        self, data: Data, mode: str = 'wb'
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
    ) -> Iterator[Shaped[np.ndarray, "..."]]:
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

    def consume(
        self, stream: Streamable[Data | Sequence[Data]], thread: bool = False
    ) -> None:
        """Consume iterable or queue and write to file.

        - If `Iterable`, fetches from the iterator until exhausted (i.e. until
          `StopIteration`), then returns.
        - If `Queue`, `.get()` from the `Queue` until `None` is received, then
          return.

        Args:
            stream: stream to consume.
            thread: whether to return immediately, and run in a separate thread
                instead of returning immediately.
        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        raise NotImplementedError(
            "`.consume()` is not implemented for this channel type.")

    def stream_prefetch(
        self, transform: Optional[
            Callable[[Shaped[np.ndarray, "..."]], Any]
        ] = None, size: int = 64, batch: int = 0
    ) -> Iterator[Shaped[np.ndarray, "..."]]:
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
