"""Channel base class."""

import os
from abc import ABC
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from typing import Any, cast

import numpy as np
from jaxtyping import Shaped

from .utils import Data, Prefetch, Streamable


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
        self, path: str, dtype: str | type | np.dtype, shape: Sequence[int]
    ) -> None:
        self.path: str = path
        self.type: np.dtype = np.dtype(dtype)
        self.shape: tuple[int, ...] = tuple(shape)
        self.size: int = int(np.prod(shape) * np.dtype(self.type).itemsize)

    def open_like(self, path: str) -> "Channel":
        """Open a channel with the same metadata, but different data."""
        return self.__class__(path=path, dtype=self.type, shape=self.shape)

    def buffer_to_array(
        self, data: bytes | bytearray, batch: bool = True, final: bool = True
    ) -> Shaped[np.ndarray, "n ..."]:
        """Convert raw buffer to the appropriate type and shape.

        !!! warning

            If `data` is a `bytes`, this method will create a new mutable
            copy from it unless `final=False`.

        Args:
            data: input buffer; use `bytearray` if downstream code requires
                a writable array.
            batch: whether this is supposed to be a batch of samples.
            final: whether the result will be directly passed to the user, or
                if a copy will be made later.

        Returns:
            Array with the appropriate type and shape, with a leading axis
                corresponding to the number of samples, if `batch=True`.
        """
        if isinstance(data, bytes):
            data = bytearray(data)

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

    def __getitem__(
        self, index: int | np.integer
    ) -> Shaped[np.ndarray, "..."]:
        """Read a single sample, i.e., `.read(start=index, samples=1)`."""
        return self.read(start=index, samples=1)

    def read(
        self, start: int | np.integer = 0, samples: int | np.integer = -1
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
    ) -> bytes | bytearray | memoryview:
        """Serialize data into a binary format for writing."""
        if isinstance(data, (list, tuple)):
            return b''.join(self._serialize(x) for x in data)

        data = cast(Data, data)
        if isinstance(data, np.ndarray):
            return data.data
        else:
            return data

    def write(
        self, data: Data, append: bool = False
    ) -> None:
        """Write data.

        Args:
            data: data to write.
            append: if `True`, append to the file instead of overwriting it.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        raise NotImplementedError(
            "`.write()` is not implemented for this channel type.")

    def stream(
        self, transform: Callable[
            [Shaped[np.ndarray, "..."]], Any] | None = None, batch: int = 0
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
        self, transform: Callable[
            [Shaped[np.ndarray, "..."]], Any] | None = None,
        size: int = 64, batch: int = 0
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
        return "{}({}: {} x {})".format(
            self.__class__.__name__, self.path, str(self.type), self.shape)
