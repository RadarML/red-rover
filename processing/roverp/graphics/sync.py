"""Video sychronization utilities."""

import numpy as np
from beartype.typing import Any, Iterator, Optional
from jaxtyping import Float


def synchronize(
    streams: dict[str, Iterator[Any]],
    timestamps: dict[str, Float[np.ndarray, "?T"]],
    period: float = 1 / 30.0,
    round: Optional[float] = None,
    duplicate: dict[str, str] = {},
    batch: int = 0,
    stop_at: float = 0.0
) -> Iterator[
    tuple[float, dict[str, int], Any]
    | list[tuple[float, dict[str, int], Any]]
]:
    """Sychronize asynchronous video/data streams.

    Args:
        streams: input iterator streams to synchronize.
        timestamps: timestamp arrays for each stream.
        period: query period, in seconds.
        round: if specified, round the start time up to the nearest `round`
            seconds.
        duplicate: duplicate selected streams (values) into the specified keys.
        batch: batch size for parallelized pipelines. If `batch=0` (default),
            no batching is applied.
        stop_at: terminate early after this many seconds. If `0.0` (default),
            plays back the full provided streams.

    Returns:
        Yields a tuple with the current timestamp relative to the start time,
        the index of each synchronized frame, and references to the active
        values at that timestamp. The active values are given by reference
        only, and should not be modified.
    """
    def handle_duplicates(t, ii, active):
        for k, v in duplicate.items():
            ii[k] = ii[v]
            active[k] = active[v]
        return t, ii, active

    if set(streams.keys()) != set(timestamps.keys()):
        raise ValueError("Streams and timestamps do not have matching keys.")

    ii = {k: 0 for k in timestamps}
    active = {k: next(v) for k, v in streams.items()}

    start_time = max(v[0] for v in timestamps.values())
    if round is not None:
        start_time = start_time // round + round
    ts = start_time

    _batch: list = []
    try:
        while True:
            for k in timestamps:
                while timestamps[k][ii[k]] < ts:
                    ii[k] += 1
                    if ii[k] >= timestamps[k].shape[0]:
                        raise StopIteration

                    try:
                        active[k] = next(streams[k])
                    except StopIteration:
                        print(f"Exhausted: {k}")
                        raise StopIteration

            if batch == 0:
                yield handle_duplicates(ts - start_time, ii, active)
            else:
                # Need to make sure we get a copy of ii and active!
                _batch.append(handle_duplicates(
                    ts - start_time, dict(**ii), dict(**active)))
                if len(_batch) == batch:
                    yield _batch
                    _batch = []
            ts += period

            if stop_at > 0 and ts > start_time + stop_at:
                print(f"Stopping early: t=+{ts - start_time:.3f}s")
                raise StopIteration

    except StopIteration:
        pass
