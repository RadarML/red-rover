"""Timestamp interpolation."""

import numpy as np
from jaxtyping import Float64


def smooth_timestamps(
    x: Float64[np.ndarray, "n"], interval: float = 30.
) -> Float64[np.ndarray, "n"]:
    """Apply piecewise linear smoothing to system timestamps.

    Corresponds to a "independent jitter" timestamp model, where each timestamp
    is jittered by some IID noise relative to a base frequency which may
    drift over time.

    Applies to: Radar, Camera, IMU

    Args:
        x: input timestamp array.
        interval: piecewise linear interpolation interval, in seconds.

    Returns:
        Smoothed timestamp array.
    """
    blocksize = int(interval * (len(x) / (x[-1] - x[0])))
    start = 0
    out = np.zeros(x.shape, x.dtype)
    while x.shape[0] > 0:
        end = min(blocksize, x.shape[0])
        out[start:start + end] = np.linspace(x[0], x[end - 1], end)
        x = x[end:]
        start += end

    return out


def discretize_timestamps(
    x: Float64[np.ndarray, "n"], interval: float = 10.
) -> Float64[np.ndarray, "n"]:
    """Apply timestamp difference discretization.
    
    Corresponds to a "frame drop" timestamp model, where frames arrive at a
    fixed rate (which may vary slightly over time), but may be randomly dropped
    with some (small) probability.

    Applies to: Lidar

    Args:
        x: input timestamp array.
        interval: interpolation interval, in seconds.
    
    Returns:
        Discretized timestamp array.
    """
    continuous = np.diff(x)
    discrete = np.round(continuous / np.median(continuous))

    blocksize = int(interval * (len(x) / (x[-1] - x[0])))
    start = 0
    out = np.zeros(x.shape, x.dtype)
    while x.shape[0] > 0:
        end = min(blocksize, x.shape[0])

        stepsize = (x[end - 1] - x[0]) / np.sum(discrete[:end - 1])
        out[start:start + end] = np.concatenate(
            [np.array([0.]), np.cumsum(discrete[:end - 1]) * stepsize]) + x[0]

        x = x[end:]
        discrete = discrete[end:]
        start += end

    return out
