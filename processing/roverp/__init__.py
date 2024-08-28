r"""Rover data processing and loading library.
::

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform
.
"""  # noqa: D205

from .slam import Poses, Trajectory, RawTrajectory
from .radar import (
    doppler_range_azimuth, doppler_range_azimuth_elevation,
    RadarProcessing, CFAR, AOAEstimation)
from .metrics import mse, ssim
from . import graphics
from .voxelgrid import VoxelGrid

__all__ = [
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
