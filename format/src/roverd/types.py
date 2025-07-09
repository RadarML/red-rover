"""Data Type Library.

!!! info

    Following the recommendations of the [abstract dataloader](
    https://wiselabcmu.github.io/abstract-dataloader/types/), each type is a
    generic dataclass of arrays.

Each type in this library is a dataclass of arrays:

- Each entry has a leading `batch` axis, which is usually `1` when loading
    single samples. Some entries are annotated with `#batch` to indicate that
    broadcasting is allowed on this axis, e.g., all samples in the same batch
    having the same `range_resolution`.
- Each entry has a second `t` axis, which indicates the time dimension for
    sequences of frames. If only single frames are loaded, this is just `1`.
"""

from typing import Generic

from abstract_dataloader.ext.types import TArray, dataclass
from jaxtyping import (
    Complex64,
    Float,
    Float32,
    Float64,
    Int16,
    Integer,
    UInt8,
    UInt16,
)


@dataclass
class XWRRadarIQ(Generic[TArray]):
    """Raw I/Q data.

    Attributes:
        iq: raw I/Q data in slow-tx-rx-fast order; see the [`xwr` documentation](
            https://wiselabcmu.github.io/xwr/dca/types/#xwr.capture.types.RadarFrame)
            for details.
        timestamps: timestamp for each frame; nominally in seconds.
        range_resolution: range resolution for the modulation used; nominally
            in meters.
        doppler_resolution: doppler resolution; nominally in m/s.
        valid: whether the entire frame is valid, or if the frame contains some
            zero-filled data due to dropped packets.
    """

    iq: Int16[TArray, "batch t slow tx rx fast"]
    timestamps: Float64[TArray, "batch t"]
    range_resolution: Float[TArray, "#batch"]
    doppler_resolution: Float[TArray, "#batch"]
    valid: UInt8[TArray, "batch t"]


@dataclass
class XWR4DSpectrum(Generic[TArray]):
    """4D radar spectrum data.

    Attributes:
        spectrum: 4D complex spectrum in doppler-elevation-azimuth-range order.
        timestamps: timestamp for each frame; nominally in seconds.
        range_resolution: range resolution for the modulation used; nominally
            in meters.
        doppler_resolution: doppler resolution; nominally in m/s.
    """

    spectrum: Complex64[TArray, "batch t doppler elevation azimuth range"]
    timestamps: Float64[TArray, "batch t"]
    range_resolution: Float[TArray, "#batch"]
    doppler_resolution: Float[TArray, "#batch"]


@dataclass
class Depth(Generic[TArray]):
    """Generic raw depth data.

    Attributes:
        rng: raw range measurements in beam-time space; nominally in mm.
        timestamps: timestamp for each frame; nominally in seconds.
    """

    rng: UInt16[TArray, "batch t beam time"]
    timestamps: Float64[TArray, "batch t"]


@dataclass
class PointCloud(Generic[TArray]):
    """Generic point cloud data.

    Attributes:
        xyz: point cloud coordinates in meters.
        length: number of points in each point cloud.
        timestamps: timestamp for each frame; nominally in seconds.
    """

    xyz: Float32[TArray, "batch t 3"]
    length: Integer[TArray, "#batch #t"]
    timestamps: Float64[TArray, "batch t"]


@dataclass
class OSDepth(Generic[TArray]):
    """Lidar raw depth data from an Ouster OSX sensor.

    !!! warning

        The measurements are recorded in *time* space, so are "staggered,"
        and must be destaggered. See the [ouster-sdk](
        https://static.ouster.dev/sdk-docs/reference/lidar-scan.html#staggering-and-destaggering)
        for details.

    Attributes:
        rng: raw range measurements in beam-time space; nominally in mm.
        timestamps: timestamp for each frame; nominally in seconds.
        intrinsics: path to intrinsics file used by the Ouster SDK.
    """

    rng: UInt16[TArray, "batch t beam time"]
    timestamps: Float64[TArray, "batch t"]
    intrinsics: str


@dataclass
class OSData(Generic[TArray]):
    """Depth and IR intensity data from an Ouster OSX sensor.

    !!! warning

        The measurements are recorded in *time* space, so are "staggered,"
        and must be destaggered. See the [ouster-sdk](
        https://static.ouster.dev/sdk-docs/reference/lidar-scan.html#staggering-and-destaggering)
        for details.

    Attributes:
        rng: raw range measurements in beam-time space; nominally in mm.
        rfl: reflectivity measurements (in NIR) for each beam (0-255).
        nir: near-infrared ambient photos.
        timestamps: timestamp for each frame; nominally in seconds.
        intrinsics: path to intrinsics file used by the Ouster SDK.
    """

    rng: UInt16[TArray, "batch t beam time"]
    rfl: UInt8[TArray, "batch t beam time"]
    nir: UInt16[TArray, "batch t beam time"]
    timestamps: Float64[TArray, "batch t"]
    intrinsics: str


@dataclass
class CameraData(Generic[TArray]):
    """Video data.

    Attributes:
        image: raw image data in HWC format.
        timestamps: timestamp for each frame; nominally in seconds.
    """

    image: UInt8[TArray, "batch t height width 3"]
    timestamps: Float64[TArray, "batch t"]


@dataclass
class CameraSemseg(Generic[TArray]):
    """Camera-based semantic segmentation.

    Attributes:
        semseg: segmentation classes.
        timestamps: timestamp for each frame; nominally in seconds.
    """

    semseg: UInt8[TArray, "batch t height width"]
    timestamps: Float64[TArray, "batch t"]


@dataclass
class IMUData(Generic[TArray]):
    """IMU data.

    Attributes:
        acc: linear acceleration, in m/s^2.
        rot: orientation (as euler angles).
        avel: angular velocity.
        timestamps: timestamp for each measurement; nominally in seconds.
    """

    acc: Float32[TArray, "batch t 3"]
    rot: Float32[TArray, "batch t 3"]
    avel: Float32[TArray, "batch t 3"]
    timestamps: Float64[TArray, "batch t"]


@dataclass
class Pose(Generic[TArray]):
    """Pose data.

    Attributes:
        pos: position, in meters (front-left-up coordinate convention).
        vel: velocity, in meters/second.
        acc: acceleration, in meters/second^2.
        rot: rotation (as matrix).
        timestamps: timestamp for each measurement; nominally in seconds.
    """

    pos: Float32[TArray, "batch t 3"]
    vel: Float32[TArray, "batch t 3"]
    acc: Float32[TArray, "batch t 3"]
    rot: Float32[TArray, "batch t 3 3"]
    timestamps: Float64[TArray, "batch t"]
