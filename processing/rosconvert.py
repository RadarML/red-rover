import os
import lzma
from tqdm import tqdm

import numpy as np
from scipy.spatial.transform import Rotation

from ouster import client
from rosbags.rosbag1 import Writer
from rosbags.serde import cdr_to_ros1, serialize_cdr
from rosbags.typesys import types



dataset = "../collect/data/test.2024.02.28-16.44.47"

with Writer('test.bag') as writer:
    topic = '/imu'
    msgtype = types.sensor_msgs__msg__Imu.__msgtype__
    connection = writer.add_connection(topic, msgtype, latching=True)
    path = os.path.join(dataset, "imu")

    with open(os.path.join(path, "acc"), 'rb') as f:
        _acc = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 3)

    with open(os.path.join(path, "avel"), 'rb') as f:
        _avel = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 3)

    with open(os.path.join(path, "rot"), 'rb') as f:
        _rot3 = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 3)
        _rot = Rotation.from_euler('XYZ', _rot3).as_quat()

    with open(os.path.join(path, "ts"), 'rb') as f:
        ts_float = np.frombuffer(f.read(), dtype=np.float64)
        _ts_sec = ts_float.astype(np.int64)
        _ts_nsec = ((ts_float - _ts_sec) * 1e9).astype(np.int64)

    NULL_COV = np.zeros(9, dtype=np.float64)

    for i in tqdm(range(ts_float.shape[0])):
        ts = types.builtin_interfaces__msg__Time(
            sec=_ts_sec[i], nanosec=_ts_nsec[i])
        header = types.std_msgs__msg__Header(frame_id="/local/imu", stamp=ts)

        rot = types.geometry_msgs__msg__Quaternion(
            x=_rot[i][0], y=_rot[i][1], z=_rot[i][2], w=_rot[i][3])
        avel = types.geometry_msgs__msg__Vector3(
            x=_avel[i][0], y=_avel[i][1], z=_avel[i][2])
        acc = types.geometry_msgs__msg__Vector3(
            x=_acc[i][0], y=_acc[i][1], z=_acc[i][2])

        imu = types.sensor_msgs__msg__Imu(
            header=header, 
            orientation=rot, orientation_covariance=NULL_COV, 
            angular_velocity=avel, angular_velocity_covariance=NULL_COV,
            linear_acceleration=acc, linear_acceleration_covariance=NULL_COV)

        writer.write(
            connection, _ts_sec[i] * 1000 * 1000 * 1000 + _ts_nsec[i],
            cdr_to_ros1(serialize_cdr(imu, msgtype), msgtype))


    topic = '/points2'
    msgtype = types.sensor_msgs__msg__PointCloud2.__msgtype__
    connection = writer.add_connection(topic, msgtype, latching=True)
    path = os.path.join(dataset, "lidar")

    fields = [
        types.sensor_msgs__msg__PointField(
            name=d, offset=4 * i,
            datatype=types.sensor_msgs__msg__PointField.FLOAT32, count=1)
        for i, d in enumerate("xyz")]

    with open(os.path.join(path, "lidar.json")) as f:
        lut = client.XYZLut(client.SensorInfo(f.read()))

    with open(os.path.join(path, "ts"), 'rb') as f:
        frametime = np.frombuffer(f.read(), dtype=np.float64)

    with lzma.open(os.path.join(path, "time"), 'rb') as f:
        ts_float = np.frombuffer(f.read(), dtype=np.float64).reshape(-1, 2048)
        ts_float = ts_float - ts_float[:, 0][:, None] + frametime[:, None]
        _ts_sec = ts_float.astype(np.int64)
        _ts_nsec = ((ts_float - _ts_sec) * 1e9).astype(np.int64)

    rng = lzma.open(os.path.join(path, "rng"))

    for i in tqdm(range(frametime.shape[0])):

        frame = np.frombuffer(
            rng.read(64 * 2048 * 2), dtype=np.uint16).reshape(64, 2048)
        points = lut(frame)
        for col in range(2048):
            ts = types.builtin_interfaces__msg__Time(
                sec=_ts_sec[i, col], nanosec=_ts_nsec[i, col])
            header = types.std_msgs__msg__Header(frame_id="/local/lidar", stamp=ts)

            data = np.frombuffer(points[:, i].tobytes(), dtype=np.uint8)

            lidar = types.sensor_msgs__msg__PointCloud2(
                header=header, height=1, width=64, fields=fields, is_bigendian=False,
                point_step=3 * 4, row_step=3 * 4, data=data, is_dense=True)

            writer.write(
                connection,
                _ts_sec[i, col] * 1000 * 1000 * 1000 + _ts_nsec[i, col],
                cdr_to_ros1(serialize_cdr(lidar, msgtype), msgtype))
