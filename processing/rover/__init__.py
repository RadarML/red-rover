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
from .dataset import (SensorData, LidarData, Dataset)

__all__ = [
    "Prefetch", "BaseChannel", "RawChannel", "LzmaChannel", "CHANNEL_TYPES",
    "SensorData", "LidarData", "Dataset", 
]
