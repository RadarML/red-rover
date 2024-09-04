"""Sensor types."""

from ._timestamps import discretize_timestamps, smooth_timestamps
from .base import SensorData
from .lidar import LidarData
from .radar import RadarData

SENSOR_TYPES: dict[str, type[SensorData]] = {
    "lidar": LidarData,
    "_lidar": LidarData,
    "radar": RadarData,
}

__all__ = [
    "SensorData", "LidarData", "RadarData",
    "smooth_timestamps", "discretize_timestamps"
]
