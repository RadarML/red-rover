"""Common capture utilities."""

import os
import json
import time
import struct
from time import perf_counter

import numpy as np

from beartype.typing import Callable


class BaseCapture:
    """Capture data for a generic sensor stream."""

    _STATS: dict[str, Callable[[np.ndarray], float]] = {
        "mean": np.mean,
        "p1": lambda x: np.percentile(x, 1),
        "p50": lambda x: np.percentile(x, 50),
        "p99": lambda x: np.percentile(x, 99)
    }

    def __init__(
        self, path: str, meta: dict[dict[str, any]], fps: float = 1.0
    ) -> None:
        os.makedirs(path)
        self.len = 0

        with open(os.path.join(path, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=4)

        self.ts = open(os.path.join(path, "ts"), 'wb')
        self.util = open(os.path.join(path, "util"), 'wb')
        self.period: list[float] = []
        self.runtime: list[float] = []
        self.prev_time = self.start_time = self.trace_time = perf_counter()
        self.fps = fps

    def start(self):
        """Mark start of current frame processing.

        (1) Records the current time as the timestamp for this frame, and
        (2) Marks the start of time utilization calculation for this frame.
        """
        t = time.time()

        self.start_time = perf_counter()
        self.ts.write(struct.pack('d', t))
        self.len += 1

    def end(self):
        """Mark end of current frame processing."""
        assert self.start_time > 0
        end = perf_counter()

        self.period.append(end - self.prev_time)
        self.runtime.append(end - self.start_time)
        self.prev_time = end
        self.util.write(struct.pack('f', end - self.start_time))

    def write(self, *args, **kwargs) -> None:
        """Write a single frame."""
        raise NotImplementedError()

    def close(self) -> None:
        """Close files and clean up."""
        self.ts.close()
        self.util.close()

    def reset_stats(self) -> None:
        """Reset tracked statistics."""
        period = np.array(self.period)
        runtime = np.array(self.runtime)
        print("freq: {}  util: {}".format(
            " ".join("{}={:5.2f}".format(
                k, 1 / v(period)) for k, v in self._STATS.items()),
            " ".join("{}={:5.2f}".format(
                k, self.fps * v(runtime)) for k, v in self._STATS.items())))
        self.period = []
        self.runtime = []
