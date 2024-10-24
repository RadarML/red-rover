"""LZMA-compressed channels."""

import lzma
from multiprocessing.pool import ThreadPool
from queue import Queue
from threading import Thread

import numpy as np
from beartype.typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    cast,
)
from jaxtyping import Shaped

from .base import Buffer, Channel, Data, Prefetch, Streamable, batch_iterator
from .raw import RawChannel


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
        self, data: Data, mode: str = 'wb'
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
            raise ValueError(f"Could not read indices: {self.path}_i.")

        data = []
        with open(self.path, 'rb') as f:
            f.seek(int(indices[0]), 0)
            for left, right in zip(indices[:-1], indices[1:]):
                decompressed = lzma.decompress(f.read(right - left))
                data.append(self.buffer_to_array(decompressed))

        return np.concatenate(data, axis=0)

    def write(
        self, data: Data, mode: str = 'wb', preset: int = 0
    ) -> None:
        """Write data.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("LzmaFrame does not support writing raw data.")

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

    def _consume(self, stream: Iterable[list[bytes]]) -> None:
        """Consume pre-compressed stream."""
        main = open(self.path, 'wb')
        offsets = open(self.path + "_i", 'wb')
        tail = 0
        offsets.write(np.array(tail, dtype=np.uint64).tobytes())

        for frame in stream:
            for x in frame:
                main.write(x)
                tail += len(x)
                offsets.write(np.array(tail, dtype=np.uint64).tobytes())

        main.close()
        offsets.close()

    def consume(
        self, stream: Streamable[Data | Sequence[Data]],
        thread: bool = False, preset: int = 0, batch: int = 8
    ) -> None:
        """Consume iterable or queue and write to file.

        - If `Iterable`, fetches from the iterator until exhausted (i.e. until
          `StopIteration`), then returns.
        - If `Queue`, `.get()` from the `Queue` until `None` is received, then
          return.

        Args:
            stream: stream to consume; possibly already batched (see `batch`).
            thread: whether to return immediately, and run in a separate thread
                instead of returning immediately.
            preset: lzma compression preset to use.
            batch: aggregate, then batch this many lzma compressions in
                parallel. Necessary for throughput reasons, since lzma
                is only single (?) threaded. If `batch=0`, we assume that
                the input stream is already batched.

        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        if isinstance(stream, Queue):
            stream = Buffer(stream)
        # => stream is Iterator | Iterable

        if thread:
            Thread(target=self.consume, kwargs={
                "stream": stream, "preset": preset, "batch": batch
            }).start()
            return

        if batch != 0:
            stream_not_batched = cast(Iterable[Data] | Iterator[Data], stream)
            stream = batch_iterator(stream_not_batched, size=batch)

        def _compress_batch(data: Sequence[Data]):
            self._verify_type(data)
            if not isinstance(data[0], np.ndarray):
                raise ValueError(
                    "LzmaFrame requires in put data to be a `np.ndarray`. "
                    "In particular, raw data (bytes) is not allowed.")

            with ThreadPool(processes=len(data)) as p:
                return p.map(
                    lambda x: lzma.compress(x.data, preset=preset),
                    cast(Sequence[np.ndarray], data))

        self._consume(Prefetch(
            _compress_batch(x) for x in
            cast(Iterable[Sequence[Data]] | Iterator[Sequence[Data]], stream)))
