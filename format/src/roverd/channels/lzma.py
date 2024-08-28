"""LZMA-compressed channels."""

import numpy as np
import lzma
from functools import partial
from multiprocessing.pool import ThreadPool
from jaxtyping import Shaped
from beartype.typing import Iterable, Iterator, Callable, Optional, Any

from .raw import RawChannel
from .base import Channel


class LzmaChannel(RawChannel):
    """LZMA-compressed binary data."""

    _FOPEN = staticmethod(lzma.open)  # type: ignore

    def memmap(self) -> np.memmap:
        """Open memory mapped array."""
        raise Exception("Cannot mem-map a compressed channel.")

    def read(
        self, start: int = 0, samples: int = -1
    ) -> Shaped[np.ndarray, "..."]:
        """Read data."""
        if start != 0:
            raise ValueError("Cannot seek (start != 0) in a LzmaChannel.")
        return super().read(start=0, samples=samples)

    def write(
        self, data: Shaped[np.ndarray, "..."], mode: str = 'wb'
    ) -> None:
        """Write data."""
        if mode != 'wb':
            raise ValueError("Can only write/overwrite an lzma channel.")
        super().write(data, mode='wb')


class LzmaFrameChannel(Channel):
    """Frame-wise LZMA-compressed binary data.

    Should have an additional file with the suffix `_i`, e.g. `mask`, `mask_i`
    which contains the starting offsets for each frame as a u8 (8-byte
    unsigned integer).

    This file should have the offset for the next unwritten frame as well.
    As an example, for a channel `example` with compressed frame sizes
    `[2, 5, 3]`, `example_i` should be::

        [0, 2, 7, 10].
    """

    def read(
        self, start: int = 0, samples: int = -1
    ) -> Shaped[np.ndarray, "samples ..."]:
        """Read data.

        Args:
            start: start index to read.
            samples: number of samples/frames to read. If `-1`, read all data.

        Returns:
            Read frames as an array, with a leading axis corresponding to
            the number of `samples`. If only a subset of frames are readable
            (e.g. due to reaching the end of the video), the result is
            truncated.

        Raises:
            ValueError: None of the frames could be read, possibly due to
                an invalid video, or invalid start index.
        """
        with open(self.path + "_i", "rb") as f:
            if start != 0:
                f.seek(8 * start)

            if samples == -1:
                indices = np.frombuffer(f.read(), dtype=np.uint64)
            else:
                indices = np.frombuffer(
                    f.read((samples + 1) * 8), dtype=np.uint64)

        if len(indices) < 2:
            raise ValueError("Could not read indices.")

        data = []
        with open(self.path, 'rb') as f:
            f.seek(int(indices[0]), 0)
            for left, right in zip(indices[:-1], indices[1:]):
                decompressed = lzma.decompress(f.read(right - left))
                data.append(self.buffer_to_array(decompressed))

        return np.concatenate(data, axis=0)

    def write(
        self, data: Shaped[np.ndarray, "n ..."], mode: str = 'wb',
        preset: int = 0
    ) -> None:
        """Write data.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        self._verify_type(data)
        if mode != 'wb':
            raise ValueError("Only overwriting is currently supported.")
        self.consume(data, preset=preset)

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
        if transform is None:
            transform = lambda x: x

        with open(self.path + "_i", "rb") as f:
            indices = np.frombuffer(f.read(), dtype=np.uint64)

        frames: list[np.ndarray] = []
        with open(self.path, 'rb') as f:
            for left, right in zip(indices[:-1], indices[1:]):
                if batch != 0 and len(frames) == batch:
                    yield transform(np.concatenate(frames, axis=0))
                    frames = []

                decompressed = self.buffer_to_array(
                    lzma.decompress(f.read(right - left)))
                if batch == 0:
                    yield transform(decompressed[0])
                else:
                    frames.append(decompressed)

        if len(frames) > 0:
            yield transform(np.concatenate(frames, axis=0))

    def consume(self, iterator: Iterable[np.ndarray], preset: int = 0) -> None:
        """Consume iterator and write to file.

        NOTE: if consuming batched data, compressesion is handled in parallel
        by a thread pool.

        Args:
            iterator: iterable to consume. Should yield frame-by-frame.
            preset: LZMA compression preset to use (0-9).

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        offsets = [0]
        with open(self.path, 'wb') as f:
            for data in iterator:
                self._verify_type(data)
                if len(data.shape) == len(self.shape):
                    compressed: bytes = lzma.compress(data.data, preset=preset)
                    f.write(compressed)
                    offsets.append(offsets[-1] + len(compressed))
                else:
                    batch: list[bytes] = ThreadPool(
                        processes=data.shape[0]
                    ).map(lambda x: lzma.compress(x.data, preset=preset), data)
                    for frame in batch:
                        f.write(frame)
                        offsets.append(offsets[-1] + len(frame))

        offsets_np = np.array(offsets, dtype=np.uint64)
        with open(self.path + "_i", 'wb') as f:
            f.write(offsets_np.data)
