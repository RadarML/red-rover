"""Convert to ROS 1 bag."""

import os
from typing import cast

from roverd import Trace, sensors


def cli_rosbag(
    path: str, /,
    out: str | None = None, min_range: float | None = None
) -> None:
    """Write lidar and IMU data to a ROS 1 bag.

    ```sh
    roverd rosbag data/wiselab --min_range 0.5
    ```

    !!! warning

        This CLI command requires the `roverd[ros]` extra to be installed.

    Args:
        path: data path.
        out: output rosbag file path; if `None`, uses `_scratch/lidar.bag` in
            the dataset directory.
        min_range: minimum range (in meters) for lidar points.
    """
    try:
        from roverd.transforms.ros import rover_to_rosbag
    except ImportError as e:
        raise ImportError(
            f"Could not import `rover_to_ros` ({e}). Make sure the `ros` "
            f"extra is installed (i.e., `pip install roverd[ros]`).")

    if out is None:
        out = os.path.join(path, "_scratch", "lidar.bag")

    trace = Trace.from_config(
        path, sensors={"lidar": sensors.OSLidarDepth, "imu": sensors.IMU})
    lidar = cast(sensors.OSLidarDepth, trace["lidar"])
    imu = cast(sensors.IMU, trace["imu"])

    rover_to_rosbag(out=out, lidar=lidar, imu=imu, min_range=min_range)
