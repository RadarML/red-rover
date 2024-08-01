r"""Rover data processing and loading library.
::

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform
.
"""  # noqa: D205

from .channel import (
    Prefetch, BaseChannel, RawChannel, LzmaChannel, CHANNEL_TYPES)
from .dataset import SensorData, LidarData, RadarData, Dataset
from .timestamps import smooth_timestamps, discretize_timestamps
from .slam import Poses, Trajectory, RawTrajectory
from .radar import (
    doppler_range_azimuth, doppler_range_azimuth_elevation,
    RadarProcessing, CFAR, AOAEstimation)
from .metrics import mse, ssim
from . import graphics
from .voxelgrid import VoxelGrid

__all__ = [
    # Data marshalling
    "Prefetch", "BaseChannel", "RawChannel", "LzmaChannel", "CHANNEL_TYPES",
    "SensorData", "LidarData", "RadarData", "Dataset",
    "smooth_timestamps", "discretize_timestamps",
    # Poses
    "Poses", "Trajectory", "RawTrajectory",
    # Radar processing
    "doppler_range_azimuth", "doppler_range_azimuth_elevation",
    "RadarProcessing", "CFAR", "AOAEstimation",
    # Metrics
    "mse", "ssim",
    # Graphics & visualization
    "graphics",
    "VoxelGrid"
]
