"""Convert Rover dataset to ROS 1 bag for Cartographer.

Inputs:
    - `lidar/*`

Outputs:
    - `_scratch/lidar.bag`
"""

import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
from beartype.typing import cast

from rosbags.rosbag1 import Writer
from rosbags.serde import cdr_to_ros1, serialize_cdr
from rosbags.typesys import types

from rover.dataset import Dataset, LidarData


def ts_sec_ns(t):
    """Convert float64 timestamp to seconds, ns remainder, and ns timestamp."""
    sec = t.astype(np.int64)
    nsec = ((t - sec) * 1e9).astype(np.int64)
    ts = sec * 1000 * 1000 * 1000 + nsec
    return sec, nsec, ts


# All zeros indicates unknown covariance.
NULL_COV = np.zeros(9, dtype=np.float64)


def sensor_msgs_imu(rot, acc, avel, sec, nsec):
    """Assemble sensor_msgs/Imu message."""

    ts = types.builtin_interfaces__msg__Time(sec=sec, nanosec=nsec)
    return types.sensor_msgs__msg__Imu(
        header=types.std_msgs__msg__Header(
            frame_id="imu_link", stamp=ts), 
        orientation=types.geometry_msgs__msg__Quaternion(
            x=rot[0], y=rot[1], z=rot[2], w=rot[3]),
        orientation_covariance=NULL_COV, 
        angular_velocity=types.geometry_msgs__msg__Vector3(
            x=avel[0], y=avel[1], z=avel[2]),
        angular_velocity_covariance=NULL_COV,
        linear_acceleration=types.geometry_msgs__msg__Vector3(
            x=acc[0], y=acc[1], z=acc[2]),
        linear_acceleration_covariance=NULL_COV)


# Must be called "x", "y", "z" for cartographer to recognize the channels.
POINTCLOUD2_FIELDS = [
    types.sensor_msgs__msg__PointField(
        name=d, offset=4 * i,
        datatype=types.sensor_msgs__msg__PointField.FLOAT32, count=1)
    for i, d in enumerate("xyz")]


def sensor_msgs_pointcloud2(points, sec, nsec):
    """Assembly sensor_msgs/PointCloud2 message."""
    ts = types.builtin_interfaces__msg__Time(sec=sec, nanosec=nsec)
    header = types.std_msgs__msg__Header(frame_id="os_sensor", stamp=ts)

    points = points.reshape(-1, 3)
    data = np.frombuffer(points.tobytes(), dtype=np.uint8)
    return types.sensor_msgs__msg__PointCloud2(
        header=header, height=1, width=points.shape[0],
        fields=POINTCLOUD2_FIELDS, is_bigendian=False, point_step=3 * 4,
        row_step=3 * 4, data=data, is_dense=True)


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out", default=None, help="Output path; defaults to "
        "`_scratch/lidar.bag` in the dataset folder.")


def _main(args):

    dataset = Dataset(args.path)
    if args.out is None:
        args.out = os.path.join(args.path, "_scratch", "lidar.bag")    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with Writer(args.out) as writer:

        imu = dataset["imu"]
        print(imu)

        msgtype = types.sensor_msgs__msg__Imu.__msgtype__
        connection = writer.add_connection("/imu", msgtype, latching=True)
        data = zip(
            Rotation.from_euler('XYZ', imu["rot"].read()).as_quat(False),
            imu["acc"].read(), imu["avel"].read(),
            *ts_sec_ns(imu.timestamps()))
        for rot, acc, avel, sec, nsec, ts in tqdm(data, total=len(imu)):
            msg = sensor_msgs_imu(rot, acc, avel, sec, nsec)
            writer.write(
                connection, ts,
                cdr_to_ros1(serialize_cdr(msg, msgtype), msgtype))

        lidar = cast(LidarData, dataset["lidar"])
        print(lidar)

        msgtype = types.sensor_msgs__msg__PointCloud2.__msgtype__
        connection = writer.add_connection("/points2", msgtype, latching=True)        
        data = zip(lidar.pointcloud_stream(), *ts_sec_ns(lidar.timestamps()))
        for points, sec, nsec, ts in tqdm(data, total=len(lidar)):
            msg = sensor_msgs_pointcloud2(points, sec, nsec)
            writer.write(
                connection, ts,
                cdr_to_ros1(serialize_cdr(msg, msgtype), msgtype))
