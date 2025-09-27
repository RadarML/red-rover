"""Rover Sensors.

| Name    | Class | Description | Timestamp Correction |
| ------- | ----- | ----------- | -------------------- |
| radar   | [`XWRRadar`][.] | 4D radar | IID |
| lidar   | [`OSLidarDepth`][.] | Ouster OS0/1/2 lidar depth data | IID + frame drops |
|         | [`OSLidar`][.] | Ouster lidar with reflectance and NIR |  |
| camera  | [`Camera`][.] | Generic RGB camera | IID |
| _camera | [`Semseg`][.] | Generic image semantic segmentation | IID |
| imu     | [`IMU`][.] | 3-axis accelerometer + gyroscope | IID |
| (other) | [`DynamicSensor`][.] | Generic sensor with dynamic channels | none |
"""

from collections.abc import Callable

from .camera import Camera, Semseg
from .generic import DynamicSensor, Sensor
from .lidar import LidarMetadata, OSLidar, OSLidarDepth
from .pose import IMU, Pose
from .radar import RadarMetadata, XWRRadar

SENSOR_TYPES: dict[str, type[Sensor]] = {
    "radar": XWRRadar,
    "lidar": OSLidarDepth,
    "camera": Camera,
    "_camera": Semseg,
    "imu": IMU,
    "pose": Pose
}


def from_config(
    path: str, type: str | None | Sensor | Callable[[str], Sensor]
) -> Sensor:
    """Create sensor from configuration.

    Args:
        path: File path to the sensor data.
        type: sensor, sensor constructor, name of a sensor, or `None` (in which
            case we use [`DynamicSensor`][roverd.sensors.DynamicSensor]).

    Returns:
        Initialized sensor object.
    """
    if isinstance(type, Sensor):
        return type
    elif type is None:
        return DynamicSensor(path)
    elif isinstance(type, str):
        if type not in SENSOR_TYPES:
            raise ValueError(f"Unknown sensor type: {type}")
        return SENSOR_TYPES[type](path)
    else:
        return type(path)


__all__ = [
    "DynamicSensor", "Sensor", "XWRRadar", "OSLidarDepth", "OSLidar", "IMU",
    "Pose", "Camera", "Semseg", "LidarMetadata", "RadarMetadata"
]
