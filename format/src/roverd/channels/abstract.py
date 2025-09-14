"""Channel abstract base classes."""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from multiprocessing.pool import ThreadPool
from queue import Queue
from threading import Thread
from typing import Any, cast

import numpy as np
from jaxtyping import Shaped

from .utils import Buffer, Data, Prefetch, Streamable


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
                f"shape {self.shape} @ channel:{self.path}.")
        if data.dtype != self.type:
            raise ValueError(f"Data type {data.dtype} does not match channel "
                f"type {self.type} @ channel:{self.path}.")

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


class BlobChannel(Channel, ABC):
    """Base class for blob channels.

    Blob channels are stored in folders, with files named `000000.{ext}`,
    `000001.{ext}`, etc. Note that names are assumed to be indices starting
    from 0 with a file extension.

    Args:
        path: file path.
        dtype: data type, or string name of dtype (e.g. `u1`, `f4`).
        shape: data shape.
        workers: maximum number of worker threads to use for I/O.
        length: number of blobs, potentially calculated some more efficient
            way. If `None`, will be calculated by counting files in the
            directory.

    Attributes:
        path: file path.
        type: numpy data type.
        shape: sample data shape.
        size: total file size, in bytes.
    """

    def __init__(
        self, path: str, dtype: str | type | np.dtype, shape: Sequence[int],
        workers: int = 8, length: int | None = None
    ) -> None:
        super().__init__(path, dtype, shape)

        self.workers = workers
        if length is not None:
            self._n_blobs = length
        elif os.path.exists(self.path):
            self._n_blobs = len(os.listdir(self.path))
        else:
            self._n_blobs = 0

    def _filename(self, index: int) -> str:
        return os.path.join(self.path, f"{index:06d}")

    @cached_property
    def filesize(self) -> int:
        """Get file size on disk in bytes."""
        return sum(
            os.stat(os.path.join(self.path, p)).st_size
            for p in os.listdir(self.path))

    @abstractmethod
    def _read_blob(self, index: int) -> np.ndarray:
        """Load a blob from file.

        Args:
            index: index of the blob to load.

        Returns:
            The loaded blob as a numpy array.
        """
        ...

    @abstractmethod
    def _write_blob(self, index: int, data: np.ndarray) -> None:
        """Write a blob to a file.

        Args:
            index: index of the blob to load.
            data: data to write.
        """
        ...

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
        if samples == -1:
            samples = self._n_blobs - start

        if start < 0 or start + samples > self._n_blobs:
            raise IndexError(
                f"Read indices ({start, start + samples}) out of range "
                f"(0, {self._n_blobs}).")

        with ThreadPool(int(min(self.workers, samples))) as pool:
            blobs = pool.map(self._read_blob, range(start, start + samples))

        return np.stack(blobs, axis=0)

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
        if batch <= 0:
            # Single sample mode
            for i in range(self._n_blobs):
                data = self.read(i, 1)[0]  # Remove batch dimension
                if transform is not None:
                    data = transform(data)
                yield data
        else:
            # Batch mode
            for start in range(0, self._n_blobs, batch):
                samples_to_read = min(batch, self._n_blobs - start)
                data = self.read(start, samples_to_read)
                if transform is not None:
                    data = transform(data)
                yield data

    def write(self, data: Data, append: bool = False) -> None:
        """Write data.

        Args:
            data: data to write, with leading axis corresponding to the number
                of samples/frames.
            append: append is currently ot implemented for blob channels.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        assert append is False, "Append is not implemented."
        if not isinstance(data, np.ndarray):
            raise ValueError("BlobChannels do not allow raw data.")
        if len(data.shape) != len(self.shape) + 1:
            raise ValueError(
                f"Data shape {data.shape} does not match channel shape "
                f"{self.shape}.")

        os.makedirs(self.path, exist_ok=True)
        with ThreadPool(self.workers) as pool:
            pool.map(
                lambda i: self._write_blob(self._n_blobs + i, data[i]),
                range(data.shape[0]))

        self._n_blobs = data.shape[0]

    def consume(
        self, stream: Streamable[Data | Sequence[Data]], thread: bool = False,
        append: bool = False
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
            append: whether to append or overwrite existing blobs.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        if isinstance(stream, Queue):
            stream = Buffer(stream)
        if thread:
            Thread(target=self.consume, kwargs={"stream": stream}).start()
            return

        if not append:
            self._n_blobs = 0

        os.makedirs(self.path, exist_ok=True)
        for frame in stream:
            if not isinstance(frame, np.ndarray):
                raise ValueError("BlobChannels do not allow raw data.")
            if len(frame.shape) != len(self.shape):
                raise ValueError(
                    "BlobChannels do not allow batched write data.")

            self._verify_type(frame)
            self._write_blob(self._n_blobs, frame)
            self._n_blobs += 1
