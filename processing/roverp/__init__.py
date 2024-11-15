r"""Rover data processing and loading library.
::

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform
.
"""  # noqa: D205

from . import graphics
from .metrics import mse, ssim
from .radar import (
    CFAR,
    AOAEstimation,
    CFARProcessing,
    RadarProcessing,
    doppler_range_azimuth,
    doppler_range_azimuth_elevation,
)
from .slam import Poses, RawTrajectory, Trajectory
from .voxelgrid import VoxelGrid

__all__ = [
    # Poses
    "Poses", "Trajectory", "RawTrajectory",
    # Radar processing
    "doppler_range_azimuth", "doppler_range_azimuth_elevation",
    "RadarProcessing", "CFAR", "AOAEstimation", "CFARProcessing",
    # Metrics
    "mse", "ssim",
    # Graphics & visualization
    "graphics",
    "VoxelGrid"
]
