r"""Rover data processing pipeline.
::
     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform
.
"""

from .channel import (
    Prefetch, BaseChannel, RawChannel, LzmaChannel, CHANNEL_TYPES)
from .dataset import (SensorData, LidarData, RadarData, Dataset)
from .slam import Poses, Trajectory
from .radar import range_doppler_azimuth, CFAR

__all__ = [
    "Prefetch", "BaseChannel", "RawChannel", "LzmaChannel", "CHANNEL_TYPES",
    "SensorData", "LidarData", "RadarData", "Dataset",
    "Poses", "Trajectory",
    "range_doppler_azimuth", "CFAR"
]
