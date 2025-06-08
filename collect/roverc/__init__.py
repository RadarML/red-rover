r"""Rover data collection platform.

Supported sensors:

- Radar: TI AWR1843 / DCA1000EVM.
- IMU: XSens MTi-3.
- Lidar: Ouster lidar.
- Camera: Any UVC camera.
"""

from .camera import Camera, CameraCapture
from .common import Capture, Sensor, SensorException
from .imu import IMU, IMUCapture
from .lidar import Lidar, LidarCapture
from .radar import Radar, RadarCapture

__all__ = [
    "Capture", "Sensor", "SensorException",
    "IMU", "IMUCapture",
    "Lidar", "LidarCapture",
    "Radar", "RadarCapture",
    "Camera", "CameraCapture"
]


SENSORS = {
    "imu": IMU,
    "lidar": Lidar,
    "radar": Radar,
    "camera": Camera
}
