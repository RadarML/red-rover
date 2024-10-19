"""Buffered video writer."""

from queue import Queue
from threading import Thread

import imageio
import numpy as np
from beartype.typing import Iterator, Optional
from jaxtyping import UInt8


def write_buffered(
    queue: Queue[Optional[UInt8[np.ndarray, "H W 3"]]], out: str,
    fps: float = 30.0, codec: str = "h264",
) -> None:
    """Write video from a queue."""

    def worker():
        writer = imageio.get_writer(out, fps=fps, codec=codec)
        while True:
            frame = queue.get()
            if frame is None:
                break
            writer.append_data(frame)
        writer.close()

    Thread(target=worker, daemon=True).start()


def write_consume(
    iter: Iterator[UInt8[np.ndarray, "H W 3"]], out: str,
    fps: float = 30.0, codec: str = "h264",
) -> None:
    """Write video from an iterator."""
    queue: Queue[Optional[UInt8[np.ndarray, "H W 3"]]] = Queue(maxsize=32)
    write_buffered(queue, out=out, fps=fps, codec=codec)

    for item in iter:
        queue.put(item)
    queue.put(None)
