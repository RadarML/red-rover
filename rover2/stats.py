"""Real time statistics."""

import numpy as np
from time import perf_counter

from beartype.typing import Callable


class RTStats:

    _stats: dict[str, Callable[[np.ndarray], float]] = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "p1": lambda x: np.percentile(x, 1),
        "p5": lambda x: np.percentile(x, 5),
        "p95": lambda x: np.percentile(x, 95),
        "p99": lambda x: np.percentile(x, 99)
    }

    def __init__(self, fps: float = 1.0) -> None:
        self.period: list[float] = []
        self.runtime: list[float] = []
        self.prev_time = perf_counter()
        self.start_time = 0.0
        self.fps = fps

    def start(self) -> None:
        self.start_time = perf_counter()

    def end(self) -> None:
        assert self.start_time > 0
        end = perf_counter()

        self.period.append(end - self.prev_time)
        self.runtime.append(end - self.start_time)
        self.prev_time = end

    def summary(self) -> dict[str, dict[str, float]]:
        period = np.array(self.period)
        runtime = np.array(self.runtime)
        return {
            "frequency": {k: 1 / v(period) for k, v in self._stats.items()},
            "utilization": {
                k: v(runtime) * self.fps for k, v in self._stats.items()}}
