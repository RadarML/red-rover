"""Raw, uncompressed binary data."""

from io import BytesIO
from queue import Queue
from threading import Thread

import numpy as np
from beartype.typing import Any, Callable, Iterator, Optional, Sequence, cast
from jaxtyping import Shaped

from .base import Buffer, Channel, Data, Streamable


class RawChannel(Channel):
    """Raw (uncompressed) data."""

    _FOPEN = staticmethod(open)

    def _open(self, path: str, mode: str) -> BytesIO:
        """Open file, and handle type assertions."""
        return cast(BytesIO, self._FOPEN(path, mode))  # type: ignore


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
        with self._open(self.path, 'rb') as f:
            if start > 0:
                f.seek(self.size * start, 0)
            size = -1 if samples == -1 else samples * self.size
            data = cast(bytes, f.read(size))
        return np.frombuffer(data, dtype=self.type).reshape(-1, *self.shape)

    def write(
        self, data: Data, mode: str = 'wb'
    ) -> None:
        """Write data.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        self._verify_type(data)
        with self._open(self.path, mode) as f:
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

        with self._open(self.path, 'rb') as fp:
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

        with self._open(self.path, 'wb') as f:
            for data in stream:
                self._verify_type(data)
                f.write(self._serialize(data))

    def memmap(self) -> np.memmap:
        """Open memory mapped array."""
        return np.memmap(
            self.path, dtype=self.type, mode='r',
            shape=(self.filesize // self.size, *self.shape))
