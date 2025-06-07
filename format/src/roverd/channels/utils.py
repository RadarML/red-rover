"""Channel utilities and types."""

from queue import Queue
from threading import Thread
from typing import Generic, Iterable, Iterator, TypeVar

import numpy as np

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
