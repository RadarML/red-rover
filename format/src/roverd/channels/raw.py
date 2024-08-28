"""Raw, uncompressed binary data."""

import numpy as np
from jaxtyping import Shaped
from beartype.typing import cast, Callable, Optional, Any, Iterable, Iterator

from .base import Channel


class RawChannel(Channel):
    """Raw (uncompressed) data."""

    _FOPEN = staticmethod(open)

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
        with self._FOPEN(self.path, 'rb') as f:  # type: ignore
            if start > 0:
                f.seek(self.size * start, 0)  # type: ignore
            size = -1 if samples == -1 else samples * self.size
            data = cast(bytes, f.read(size))
        return np.frombuffer(data, dtype=self.type).reshape(-1, *self.shape)

    def write(
        self, data: Shaped[np.ndarray, "..."], mode: str = 'wb'
    ) -> None:
        """Write data.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        self._verify_type(data)
        with self._FOPEN(self.path, mode) as f:  # type: ignore
            f.write(data.tobytes())  # type: ignore

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
        shape = self.shape if batch == 0 else [batch, *self.shape]
        size = self.size if batch == 0 else batch * self.size

        if transform is None:
            transform = lambda x: x

        with self._FOPEN(self.path, 'rb') as fp:  # type: ignore
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

    def consume(self, iterator: Iterable[np.ndarray]) -> None:
        """Consume iterator and write to file.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        with self._FOPEN(self.path, 'wb') as f:  # type: ignore
            for data in iterator:
                self._verify_type(data)
                f.write(data.tobytes())  # type: ignore

    def memmap(self) -> np.memmap:
        """Open memory mapped array."""
        return np.memmap(
            self.path, dtype=self.type, mode='r',
            shape=(self.filesize // self.size, *self.shape))
