"""Video sychronization utilities."""

import numpy as np
from beartype.typing import Any, Iterator, Optional
from jaxtyping import Float


def synchronize(
    streams: dict[str, Iterator[Any]],
    timestamps: dict[str, Float[np.ndarray, "?T"]],
    period: float = 1 / 30.0,
    round: Optional[float] = None,
) -> Iterator[tuple[float, dict[str, int], Any]]:
    """Sychronize asynchronous video/data streams.

    Args:
        streams: input iterator streams to synchronize.
        timestamps: timestamp arrays for each stream.
        period: query period, in seconds.
        round: if specified, round the start time up to the nearest `round`
            seconds.

    Returns:
        Yields a tuple with the current timestamp relative to the start time,
        the index of each synchronized frame, and references to the active
        values at that timestamp. The active values are given by reference
        only, and should not be modified.
    """
    if set(streams.keys()) != set(timestamps.keys()):
        raise ValueError("Streams and timestamps do not have matching keys.")

    ii = {k: 0 for k in timestamps}
    active = {k: next(v) for k, v in streams.items()}

    start_time = max(v[0] for v in timestamps.values())
    if round is not None:
        start_time = start_time // round + round
    ts = start_time

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

            yield (ts - start_time, ii, active)
            ts += period
    except StopIteration:
        pass
