"""Blob channels consisting of folders of separate files."""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from functools import cached_property
from multiprocessing.pool import ThreadPool
from queue import Queue
from threading import Thread
from typing import Any

import numpy as np
from jaxtyping import Shaped

from .base import Channel
from .utils import Buffer, Data, Streamable


class BlobChannel(Channel, ABC):
    """Base class for blob channels.

    Blob channels are stored in folders, with files named `000000.ext`,
    `000001.ext`, etc. Note that names are assumed to be indices starting from
    0 with a file extension.

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

    def write(self, data: Shaped[np.ndarray, "n ..."]) -> None:
        """Write data.

        Args:
            data: data to write, with leading axis corresponding to the number
                of samples/frames.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
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


class NPZBlobChannel(BlobChannel):
    """Blob channel consisting of `.npz` files.

    Args:
        path: file path.
        dtype: data type, or string name of dtype (e.g. `u1`, `f4`).
        shape: data shape.
        workers: maximum number of worker threads to use for I/O.
        length: number of blobs, potentially calculated some more efficient
            way. If `None`, will be calculated by counting files in the
            directory.
        compress: whether to use compression when writing `.npz` files.

    Attributes:
        path: file path.
        type: numpy data type.
        shape: sample data shape.
        size: total file size, in bytes.
    """

    def __init__(
        self, path: str, dtype: str | type | np.dtype, shape: Sequence[int],
        workers: int = 8, length: int | None = None, compress: bool = False
    ) -> None:
        super().__init__(path, dtype, shape, workers, length)
        self.compress = compress

    def _read_blob(self, index: int) -> np.ndarray:
        """Load a blob from file.

        Args:
            index: index of the blob to load.

        Returns:
            The loaded blob as a numpy array.
        """
        filename = self._filename(index) + ".npz"
        if not os.path.exists(filename):
            raise IndexError(f"Blob index {index} does not exist.")

        with np.load(filename) as data:
            return data['data']

    def _write_blob(self, index: int, data: np.ndarray) -> None:
        """Write a blob to a file.

        Args:
            index: index of the blob to load.
            data: data to write.
        """
        filename = self._filename(index) + ".npz"
        if self.compress:
            np.savez_compressed(filename, data=data)
        else:
            np.savez(filename, data=data)


class JPEGBlobChannel(BlobChannel):
    """Blob channel consisting of `.jpg` files.

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

    @cached_property
    def _cv2_module(self):
        try:
            import cv2
            return cv2
        except ImportError:
            raise ImportError(
                "Could not import cv2. `opencv-python` or "
                "`opencv-python-headless` must be installed in order to use "
                "video encoding or decoding.")

    def _read_blob(self, index: int) -> np.ndarray:
        filename = self._filename(index) + ".jpg"
        if not os.path.exists(filename):
            raise IndexError(f"Blob index {index} does not exist.")

        img = self._cv2_module.imread(
            filename, self._cv2_module.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image at index {index}.")
        return self._cv2_module.cvtColor(img, self._cv2_module.COLOR_BGR2RGB)

    def _write_blob(self, index: int, data: np.ndarray) -> None:
        filename = self._filename(index) + ".jpg"
        img = self._cv2_module.cvtColor(data, self._cv2_module.COLOR_RGB2BGR)
        if not self._cv2_module.imwrite(filename, img):
            raise ValueError(f"Could not write image at index {index}.")
