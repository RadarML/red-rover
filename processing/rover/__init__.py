r"""Rover data processing and loading library.
::
     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform
.
"""

from .channel import (
    Prefetch, BaseChannel, RawChannel, LzmaChannel, CHANNEL_TYPES)
from .dataset import (
    SensorData, LidarData, RadarData, Dataset, smooth_timestamps)
from .slam import Poses, Trajectory, RawTrajectory
from .radar import dopper_range_azimuth, CFAR
from . import graphics

__all__ = [
    "Prefetch", "BaseChannel", "RawChannel", "LzmaChannel", "CHANNEL_TYPES",
    "SensorData", "LidarData", "RadarData", "Dataset", "smooth_timestamps",
    "Poses", "Trajectory", "RawTrajectory",
    "dopper_range_azimuth", "CFAR",
    "graphics"
]
