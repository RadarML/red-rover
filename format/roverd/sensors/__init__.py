"""Sensor types."""

from .base import SensorData
from .lidar import LidarData
from .radar import RadarData

from ._timestamps import smooth_timestamps, discretize_timestamps

SENSOR_TYPES: dict[str, type[SensorData]] = {
    "lidar": LidarData,
    "_lidar": LidarData,
    "radar": RadarData,
}

__all__ = [
    "SensorData", "LidarData", "RadarData",
    "smooth_timestamps", "discretize_timestamps"
]
