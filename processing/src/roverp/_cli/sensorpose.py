"""Get interpolated poses for a specific sensor."""

import os

import numpy as np


def cli_sensorpose(
    path: str, /, sensor: str = "radar",
    smoothing: float = 500.0, threshold: float = 1.0
) -> None:
    """Get interpolated poses for a specific sensor.

    !!! io "Expected Inputs and Outputs"

        **Inputs**: `_slam/trajectory.csv`

        **Outputs**: `_{sensor}/pose.npz` depending on the specified `--sensor`, with keys:

        - `mask`: binary mask, applied to the raw sensor data along the time
        axis, which denotes valid samples for the available poses.
        - `smoothing`, `start_threshold`, `filter_size`: parameters used for
        pose interpolation.
        - `t`, `pos`, `vel`, `acc`, `rot`: pose parameters; see
            [`Poses`][roverp.readers.]

    Args:
        path: Path to the dataset.
        sensor: Sensor timestamps to interpolate for.
        smoothing: Smoothing coefficient; higher = more smooth.
        threshold: Exclude data points close to the starting point (in meters).
    """
    from roverd import Trace

    from roverp.readers import Trajectory

    cfg = {
        "smoothing": smoothing, "start_threshold": threshold,
        "filter_size": 5}

    traj = Trajectory(
        path=os.path.join(path, "_slam", "trajectory.csv"), **cfg)
    t_sensor = Trace.from_config(path)[sensor].metadata.timestamps
    poses, mask = traj.interpolate(t_sensor)

    os.makedirs(os.path.join(path, "_" + sensor), exist_ok=True)
    np.savez(
        os.path.join(path, "_" + sensor, "pose.npz"),
        **poses.as_dict(), mask=mask, **cfg)
