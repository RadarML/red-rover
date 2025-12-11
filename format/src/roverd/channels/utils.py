"""Channel utilities and types."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

Data = np.ndarray | bytes | bytearray
"""Generic writable data.

Should generally behave as follows:

- If `Shaped[np.ndarray, "..."]`, the shape and dtype are assumed to have
  semantic meaning, and are verified.
- If `bytes` or `bytearray`, we assume that the caller has already done any
  necessary binary conversion. No type or shape verification is performed.
"""


@dataclass
class ExceptionSentinel:
    """Sentinel class to indicate an exception, which should bubble up.

    !!! info

        We use an `ExceptionSentinel` wrapping the actual exception in case
        users want to put an actual `Exception` into a stream.
    """

    exception: Exception


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ReadableQueue(Protocol[T_co]):
    """Protocol for a readable queue.

    !!! info

        This is similar to `queue.Queue`, but only requires the `get` and
        `empty` methods, and is covariant in the item type.
    """

    def get(self, block: bool = True, timeout: float | None = None) -> T_co:
        ...

    def empty(self) -> bool:
        ...


Streamable = (
    Iterator[T] | Iterable[T] | ReadableQueue[T | None | ExceptionSentinel])
"""Any stream-like container.

!!! warning

    Unlike `Iterator[T]` or `Iterable[T]`, a `Streamable: Queue` may also yield
    `None` at the end of the stream or `ExceptionSentinel` if an exception
    occurs in the producer.

!!! danger

    Instead of an invariant `Queue`, a `Streamable` object uses a covariant
    [`ReadableQueue`][^.], since producers may want to put subtypes of `T` into
    the queue. Downstream users must check the union against `ReadableQueue`
    instead of `queue.Queue`.
"""


class Buffer(Generic[T]):
    """Simple queue buffer (i.e. queue to iterator) with batching.

    Accepts the following "signals":

    - `None`: indicates that the stream is complete (i.e. `StopIteration`).
    - `ExceptionSentinel`: indicates that a generic exception occurred, which
        should be re-raised in the thread which reads from the buffer.

    Args:
        queue: queue to use as a buffer. Should return `None` when the stream
            is complete (i.e. `StopIteration`).
    """

    def __init__(
        self, queue: ReadableQueue[T | None | ExceptionSentinel]
    ) -> None:
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self) -> T:
        item = self.queue.get()
        if isinstance(item, ExceptionSentinel):
            raise item.exception
        if item is None:
            raise StopIteration
        else:
            return item


class Prefetch(Buffer[T]):
    """Simple prefetch queue wrapper (i.e. iterator to queue).

    Can be used as a prefetched iterator (`for x in Prefetch(...)`), or as a
    queue (`Prefetch(...).queue`). When used as a queue, `None` is put in the
    queue to indicate that the iterator has terminated.

    !!! warning

        Any exceptions raised in the provided iterator are re-raised in the
        thread which reads the queue.

    Args:
        iterator: any python iterator; must never yield `None`.
        size: prefetch buffer size.
    """

    def __init__(
        self, iterator: Iterable[T] | Iterator[T], size: int = 64
    ) -> None:
        self.queue = Queue(maxsize=size)
        self.iterator = iterator

        Thread(target=self._prefetch, daemon=True).start()

    def _prefetch(self) -> None:
        try:
            for item in self.iterator:
                self.queue.put(item)
            self.queue.put(None)
        except Exception as e:
            self.queue.put(ExceptionSentinel(e))


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
