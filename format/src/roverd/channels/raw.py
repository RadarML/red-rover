"""Raw, uncompressed binary data."""

import io
from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterator, Optional, Sequence, cast

import numpy as np
from jaxtyping import Shaped

from .base import Channel
from .utils import Buffer, Data, Streamable


class RawChannel(Channel):
    """Raw (uncompressed) data.

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

    @staticmethod
    def _open_r(path: str) -> io.BufferedIOBase:
        return open(path, 'rb')

    @staticmethod
    def _open_w(path, append: bool = False) -> io.BufferedIOBase:
        return open(path, 'ab' if append else 'wb')

    def read(
        self, start: int | np.integer = 0, samples: int | np.integer = -1
    ) -> Shaped[np.ndarray, "..."]:
        """Read data.

        !!! info

            We read through `bytearray -> memoryview -> np.frombuffer` to
            provide a read-write buffer without requiring an additional copy.
            This is required for full functionality in downstream applications,
            e.g. [`torch.from_numpy`](
            https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html).

            Note that this is valid since the bytearray is not returned, so
            ownership is passed to the returned numpy array.

        Args:
            start: start index to read.
            samples: number of samples/frames to read. If `-1`, read all data.

        Returns:
            Read frames as an array, with a leading axis corresponding to
                the number of `samples`.
        """
        with self._open_r(self.path) as f:
            if start > 0:
                f.seek(self.size * start, 0)

            size = self.filesize if samples == -1 else samples * self.size
            buf = bytearray(size)
            f.readinto(memoryview(buf))

        return self.buffer_to_array(buf, batch=True)

    def write(self, data: Data, append: bool = False) -> None:
        """Write data.

        Args:
            data: data to write.
            append: if `True`, append to the file instead of overwriting it.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        self._verify_type(data)
        with self._open_w(self.path, append) as f:
            f.write(self._serialize(data))

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
        size = self.size if batch == 0 else batch * self.size

        if transform is None:
            transform = lambda x: x

        with self._open_r(self.path) as fp:
            while True:
                data = cast(bytes, fp.read(size))
                if len(data) < size:
                    fp.close()
                    partial_batch = len(data) // self.size
                    if partial_batch > 0:
                        yield transform(self.buffer_to_array(
                            data[:partial_batch * size], batch=(batch != 0)))
                    return
                yield transform(self.buffer_to_array(data, batch=(batch != 0)))

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
            thread: if `True`, return immediately, and run in a separate thread
                instead of returning immediately.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        if isinstance(stream, Queue):
            stream = Buffer(stream)
        if thread:
            Thread(target=self.consume, kwargs={"stream": stream}).start()
            return

        with self._open_w(self.path, append=False) as f:
            for data in stream:
                self._verify_type(data)
                f.write(self._serialize(data))

    def memmap(self) -> np.memmap:
        """Open memory mapped array."""
        return np.memmap(
            self.path, dtype=self.type, mode='r',
            shape=(self.filesize // self.size, *self.shape))
