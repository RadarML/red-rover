"""Buffered video writer."""

from queue import Queue
from threading import Thread

import imageio
import numpy as np
from beartype.typing import Iterator, Optional
from jaxtyping import Array, UInt8


def write_buffered(
    queue: Queue[Optional[UInt8[np.ndarray | Array, "*batch H W 3"]]],
    out: str, fps: float = 30.0, codec: str = "h264",
) -> None:
    """Write video from a queue of optionally batched images."""

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
    """Write video from an iterator of optionally batched images."""
    queue: Queue[
        Optional[UInt8[np.ndarray | Array, "*batch H W 3"]]
    ] = Queue(maxsize=queue_size)
    write_buffered(queue, out=out, fps=fps, codec=codec)

    for item in iter:
        queue.put(item)
    queue.put(None)
