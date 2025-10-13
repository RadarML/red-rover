"""Rover to rosbag conversion.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from roverd.transforms import ros
    ```

    You will also need to have the `ros` extra installed.
"""

import os

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.serde import cdr_to_ros1, serialize_cdr
from rosbags.typesys import types
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from roverd import sensors
from roverd.channels import utils
from roverd.transforms import ouster


def ts_sec_ns(t):
    """Convert float64 timestamp to seconds, ns remainder, and ns timestamp."""
    ts = (t * 1e9).astype(np.int64)
    sec = t.astype(np.int64)
    nsec = ts % (1000 * 1000 * 1000)
    return sec, nsec, ts



def __sensor_msgs_imu(rot, acc, avel, sec, nsec):
    # All zeros indicates unknown covariance.
    NULL_COV = np.zeros(9, dtype=np.float64)

    ts = types.builtin_interfaces__msg__Time(sec=sec, nanosec=nsec)
    return types.sensor_msgs__msg__Imu(
        header=types.std_msgs__msg__Header(
            frame_id="imu_link", stamp=ts),
        orientation=types.geometry_msgs__msg__Quaternion(
            x=rot[0], y=rot[1], z=rot[2], w=rot[3]),
        orientation_covariance=NULL_COV,  # type: ignore
        angular_velocity=types.geometry_msgs__msg__Vector3(
            x=avel[0], y=avel[1], z=avel[2]),
        angular_velocity_covariance=NULL_COV,  # type: ignore
        linear_acceleration=types.geometry_msgs__msg__Vector3(
            x=acc[0], y=acc[1], z=acc[2]),
        linear_acceleration_covariance=NULL_COV)  # type: ignore


# Must be called "x", "y", "z" for cartographer to recognize the channels.
__POINTCLOUD2_FIELDS = [
    types.sensor_msgs__msg__PointField(
        name=d, offset=4 * i,
        datatype=types.sensor_msgs__msg__PointField.FLOAT32, count=1)
    for i, d in enumerate("xyz")]


def __sensor_msgs_pointcloud2(points, sec, nsec):
    ts = types.builtin_interfaces__msg__Time(sec=sec, nanosec=nsec)
    header = types.std_msgs__msg__Header(frame_id="os_sensor", stamp=ts)

    data = np.frombuffer(points.tobytes(), dtype=np.uint8)
    return types.sensor_msgs__msg__PointCloud2(
        header=header, height=1, width=points.shape[0],
        fields=__POINTCLOUD2_FIELDS, is_bigendian=False, point_step=3 * 4,
        row_step=3 * 4, data=data, is_dense=True)  # type: ignore


def rover_to_rosbag(
    out: str, imu: sensors.IMU, lidar: sensors.OSLidarDepth,
    min_range: float | None = None
) -> None:
    """Write lidar and IMU data to a ros bag (to run SLAM, e.g., cartographer).

    Args:
        out: output rosbag file path.
        imu: IMU sensor.
        lidar: Lidar sensor.
        min_range: minimum range for lidar points.
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with Writer(out) as writer:

        msgtype = types.sensor_msgs__msg__Imu.__msgtype__
        connection = writer.add_connection("/imu", msgtype, latching=1)
        data = zip(
            Rotation.from_euler('XYZ', imu["rot"].read()).as_quat(False),
            imu["acc"].read(), imu["avel"].read(),
            *ts_sec_ns(imu.metadata.timestamps))
        for rot, acc, avel, sec, nsec, ts in tqdm(data, total=len(imu), desc="imu"):
            msg = __sensor_msgs_imu(rot, acc, avel, sec, nsec)
            writer.write(
                connection, ts,
                cdr_to_ros1(serialize_cdr(msg, msgtype), msgtype))  # type: ignore

        to_pointcloud = ouster.PointCloud(min_range=min_range)
        msgtype = types.sensor_msgs__msg__PointCloud2.__msgtype__
        connection = writer.add_connection("/points2", msgtype, latching=1)
        data = utils.Prefetch(
            zip(lidar.stream(), *ts_sec_ns(lidar.metadata.timestamps)))
        for raw, sec, nsec, ts in tqdm(data, total=len(lidar), desc="lidar"):
            points = to_pointcloud(raw).xyz
            msg = __sensor_msgs_pointcloud2(points, sec, nsec)
            writer.write(
                connection, ts,
                cdr_to_ros1(serialize_cdr(msg, msgtype), msgtype))  # type: ignore
