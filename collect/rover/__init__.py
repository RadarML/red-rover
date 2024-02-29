r"""Rover data collection platform.
::
     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform
.

Supported sensors:
- Radar: TI AWR1843 / DCA1000EVM; full api implemented in `rover.radar_api`.
- IMU: XSens MTi-3; partial api implemented in `rover.imu_api`.
- Lidar: Ouster lidar; via the Ouster SDK.
- Camera: Any UVC camera.
"""

from .imu import IMU
from .lidar import Lidar
from .radar import Radar
from .camera import Camera
from .common import BaseSensor


__all__ = ["BaseSensor", "IMU", "Lidar", "Radar", "Camera"]


SENSORS = {
    "imu": IMU,
    "lidar": Lidar,
    "radar": Radar,
    "camera": Camera
}
