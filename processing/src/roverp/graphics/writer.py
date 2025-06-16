"""Buffered video writer."""

from queue import Queue
from threading import Thread

import imageio
import numpy as np
from beartype.typing import Iterator
from jaxtyping import Array, UInt8


def write_buffered(
    queue: Queue[UInt8[np.ndarray | Array, "*batch H W 3"] | None],
    out: str, fps: float = 30.0, codec: str = "h264",
) -> None:
    """Write video from a queue of optionally batched images.

    Args:
        queue: input queue; `None` indicates the end of the stream.
        out: output file path.
        fps: frames per second.
        codec: video codec to use; see [supported codecs](
            https://imageio.readthedocs.io/en/stable/format_gif.html#supported-codecs).
    """

    def worker():
        writer = imageio.get_writer(out, fps=fps, codec=codec)
        while True:
            frame = queue.get()
            if frame is None:
                break
            if len(frame.shape) == 4:
                for x in frame:
                    writer.append_data(np.array(x))
            else:
                writer.append_data(np.array(frame))
        writer.close()

    Thread(target=worker, daemon=True).start()


def write_consume(
    iter: Iterator[UInt8[np.ndarray | Array, "*batch H W 3"]], out: str,
    fps: float = 30.0, codec: str = "h264", queue_size: int = 32
) -> None:
    """Write video from an iterator of optionally batched images.

    Args:
        iter: input iterator.
        out: output file path.
        fps: frames per second.
        codec: video codec to use; see [supported codecs](
            https://imageio.readthedocs.io/en/stable/format_gif.html#supported-codecs).
        queue_size: maximum size of the internal queue to use for buffering.
    """
    queue: Queue[UInt8[np.ndarray | Array, "*batch H W 3"] | None]
    queue = Queue(maxsize=queue_size)
    write_buffered(queue, out=out, fps=fps, codec=codec)

    for item in iter:
        queue.put(item)
    queue.put(None)
