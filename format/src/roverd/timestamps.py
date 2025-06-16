"""Timestamp denoising / dejittering.

Timestamps which are recorded when a sample is read by the collection computer
may suffer from jitter due to scheduling noise and other timing effects.
Assuming the sensor clock is *locally* stable, we can interpolate or smooth the
timestamps to reduce this jitter.

This module provides two timestamp models:

- `smooth`: Each timestamp is jittered by some IID noise.
- `discretize`: Frames arrive at a fixed rate with minor discrepancies, but may
    be randomly dropped.
"""

import numpy as np
from jaxtyping import Float64


def identity(x: Float64[np.ndarray, "n"]) -> Float64[np.ndarray, "n"]:
    """Do not apply any correction."""
    return x


def smooth(
    x: Float64[np.ndarray, "n"], interval: float = 30.
) -> Float64[np.ndarray, "n"]:
    """Apply piecewise linear smoothing to system timestamps.

    Corresponds to a "independent jitter" timestamp model, where each timestamp
    is jittered by some IID noise relative to a base sampling frequency which
    may drift slightly over time.

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


def discretize(
    x: Float64[np.ndarray, "n"], interval: float = 10., eps: float = 0.05
) -> Float64[np.ndarray, "n"]:
    """Apply timestamp difference discretization.

    Corresponds to a "frame drop" timestamp model, where frames arrive at a
    fixed rate (which may vary slightly over time), but may be randomly dropped
    with some (small) probability.

    Applies to: Lidar

    !!! note

        The interval is always assumed to be at least one time step:
        if the received time difference is less than half a time step, it is
        rounded up to one time step.

    Args:
        x: input timestamp array.
        interval: interpolation interval, in seconds.
        eps: only rounds up (assuming a frame drop) if the interframe time
            exceeds `1.5 + eps` time steps.

    Returns:
        Discretized timestamp array.
    """
    continuous = np.diff(x)
    global_step = np.median(continuous)
    discrete = np.where(
        continuous / global_step > 1.5 + eps,
        np.round(continuous / np.median(continuous)),
        1.0)

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
