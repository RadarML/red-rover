"""Ouster Lidar transforms.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:

    ```python
    from roverd.transforms import ouster
    ```

This is a pure-numpy reimplementation of the subset of the `ouster-sdk`
lidar projection API (`SensorInfo`, `destagger`, `XYZLut`) required by
[`Destagger`][roverd.transforms.ouster.Destagger] and
[`PointCloud`][roverd.transforms.ouster.PointCloud], since `ouster-sdk` is
an unstable dependency which we would like to avoid requiring.

The projection math follows the "Lidar Range Data to Cartesian Coordinates"
algorithm described in the Ouster Software User Guide, and has been
validated to reproduce `ouster.sdk.client.XYZLut` bit-for-bit (up to
floating point error) on the sensor metadata this codebase generates.
"""

import json
from dataclasses import dataclass

import numpy as np
from abstract_dataloader import spec
from jaxtyping import Float64, Int64

from roverd import types


@dataclass
class LidarIntrinsics:
    """Parsed Ouster lidar sensor metadata (`lidar.json`).

    Only the fields required for destaggering and XYZ projection are kept;
    see the [Ouster sensor metadata reference](
    https://static.ouster.dev/sdk-docs/reference/sensor-data.html#sensor-metadata)
    for the full schema.

    Attributes:
        pixels_per_column: number of beams.
        columns_per_frame: number of measurements ("ticks") per rotation.
        pixel_shift_by_row: per-beam staggering offset, in columns.
        beam_altitude_angles: per-beam altitude angle, in degrees.
        beam_azimuth_angles: per-beam azimuth firing offset, in degrees.
        beam_to_lidar_transform: `4x4` transform from the per-beam origin to
            the lidar frame (at zero encoder angle).
        lidar_to_sensor_transform: `4x4` transform from the lidar frame to
            the sensor frame.
    """

    pixels_per_column: int
    columns_per_frame: int
    pixel_shift_by_row: Int64[np.ndarray, "beam"]
    beam_altitude_angles: Float64[np.ndarray, "beam"]
    beam_azimuth_angles: Float64[np.ndarray, "beam"]
    beam_to_lidar_transform: Float64[np.ndarray, "4 4"]
    lidar_to_sensor_transform: Float64[np.ndarray, "4 4"]

    @classmethod
    def from_json(cls, cfg: str) -> "LidarIntrinsics":
        """Parse lidar intrinsics from a `lidar.json` metadata string.

        Args:
            cfg: `lidar.json` file contents, as written by
                `ouster.sdk.client.SensorInfo.to_json_string`.

        Returns:
            Parsed lidar intrinsics.
        """
        meta = json.loads(cfg)
        fmt = meta["lidar_data_format"]
        beam = meta["beam_intrinsics"]
        lidar = meta["lidar_intrinsics"]

        return cls(
            pixels_per_column=fmt["pixels_per_column"],
            columns_per_frame=fmt["columns_per_frame"],
            pixel_shift_by_row=np.array(
                fmt["pixel_shift_by_row"], dtype=np.int64),
            beam_altitude_angles=np.array(
                beam["beam_altitude_angles"], dtype=np.float64),
            beam_azimuth_angles=np.array(
                beam["beam_azimuth_angles"], dtype=np.float64),
            beam_to_lidar_transform=np.array(
                beam["beam_to_lidar_transform"], dtype=np.float64
            ).reshape(4, 4),
            lidar_to_sensor_transform=np.array(
                lidar["lidar_to_sensor_transform"], dtype=np.float64
            ).reshape(4, 4))


class ConfigCache:
    """Ouster sensor intrinsics cache."""

    def __init__(self) -> None:
        self._by_path: dict[str, LidarIntrinsics] = {}
        self._by_cfg: dict[str, LidarIntrinsics] = {}

    def __getitem__(self, path: str) -> LidarIntrinsics:
        """Get sensor intrinsics by path.

        - First checks if we have `LidarIntrinsics` loaded already for the
            given path.
        - If not, we then check if we have `LidarIntrinsics` loaded with an
            identical configuration.
        - If neither is found, we read the file and parse a new
            `LidarIntrinsics`.

        Args:
            path: Path to the sensor intrinsics file.

        Returns:
            Parsed lidar intrinsics.
        """
        # We've never seen this path before. Need to inspect it.
        if path not in self._by_path:
            try:
                with open(path) as f:
                    cfg = f.read()
            except FileNotFoundError as e:
                raise ValueError(
                    f"Lidar sensor intrinsics {path} does not exist.") from e

            # Truly new -> load
            if cfg not in self._by_cfg:
                self._by_cfg[cfg] = LidarIntrinsics.from_json(cfg)
            # Make a link
            self._by_path[path] = self._by_cfg[cfg]

        return self._by_path[path]


class Destagger(spec.Transform[types.OSDepth, types.Depth]):
    """Destagger Ouster Lidar depth data.

    Args:
        config: if provided, use this configuration cache (to share with other
            transforms).
    """

    def __init__(self, config: ConfigCache | None = None) -> None:
        if config is None:
            config = ConfigCache()
        self._config = config

    def __call__(self, data: types.OSDepth) -> types.Depth:
        """Destagger Ouster Lidar depth data.

        Args:
            data: Ouster Lidar depth data with staggered measurements.

        Returns:
            Depth data with destaggered measurements.
        """
        info = self._config[data.intrinsics]
        shifts = info.pixel_shift_by_row
        width = data.rng.shape[-1]

        # Each beam's column is circularly shifted by its own azimuth
        # firing offset, so that all beams end up aligned to a common
        # azimuth per column: destaggered[..., h, w] = rng[..., h, (w -
        # shifts[h]) % width].
        columns = (np.arange(width)[None, :] - shifts[:, None]) % width
        destaggered = np.take_along_axis(
            data.rng, columns[None, None], axis=-1)

        return types.Depth(rng=destaggered, timestamps=data.timestamps)


@dataclass
class _LidarXYZLut:
    """Precomputed lookup table for Ouster range-to-XYZ conversion.

    Attributes:
        dx: X direction cosine, per (beam, tick).
        dy: Y direction cosine, per (beam, tick).
        dz: Z direction cosine, per beam.
        cos_enc: cosine of encoder angle, per tick.
        sin_enc: sine of encoder angle, per tick.
        beam_r: radial distance from the lidar origin to the beam origin,
            in mm.
        bx: beam origin X offset, in mm (from `beam_to_lidar_transform`).
        bz: beam origin Z offset, in mm (from `beam_to_lidar_transform`).
        rotation: rotation part of `lidar_to_sensor_transform`.
        translation: translation part of `lidar_to_sensor_transform`, in m.
    """

    dx: Float64[np.ndarray, "beam tick"]
    dy: Float64[np.ndarray, "beam tick"]
    dz: Float64[np.ndarray, "beam 1"]
    cos_enc: Float64[np.ndarray, "1 tick"]
    sin_enc: Float64[np.ndarray, "1 tick"]
    beam_r: float
    bx: float
    bz: float
    rotation: Float64[np.ndarray, "3 3"]
    translation: Float64[np.ndarray, "3"]

    @classmethod
    def from_intrinsics(cls, intrinsics: LidarIntrinsics) -> "_LidarXYZLut":
        """Precompute the XYZ lookup table from lidar intrinsic calibration.

        Args:
            intrinsics: lidar intrinsic calibration parameters.
        """
        blt = intrinsics.beam_to_lidar_transform
        bx = float(blt[0, 3])
        bz = float(blt[2, 3])

        alt = np.deg2rad(intrinsics.beam_altitude_angles)
        az_off = np.deg2rad(intrinsics.beam_azimuth_angles)

        width = intrinsics.columns_per_frame
        # Encoder angle decreases with column index (sensor rotates
        # clockwise as viewed from above).
        theta_enc = 2 * np.pi * (1.0 - np.arange(width) / width)
        # Total beam azimuth = encoder angle minus beam azimuth offset.
        az = theta_enc[None, :] - az_off[:, None]
        phi = alt[:, None]

        lts = intrinsics.lidar_to_sensor_transform

        return cls(
            dx=np.cos(phi) * np.cos(az),
            dy=np.cos(phi) * np.sin(az),
            dz=np.sin(phi) * np.ones((len(alt), 1)),
            # Beam origin XY follows the encoder angle, not the total beam
            # azimuth (the "beam_to_lidar_transform" Y offset is unused,
            # matching ouster-sdk).
            cos_enc=np.cos(theta_enc)[None, :],
            sin_enc=np.sin(theta_enc)[None, :],
            beam_r=float(np.hypot(bx, bz)),
            bx=bx,
            bz=bz,
            rotation=lts[:3, :3],
            translation=lts[:3, 3] * 1e-3)

    def compute_xyz(
        self, r: np.ndarray
    ) -> Float64[np.ndarray, "... beam tick 3"]:
        """Convert range values to XYZ coordinates in the sensor frame.

        Args:
            r: range in mm, shape `(..., beam, tick)`.

        Returns:
            XYZ in meters; zero where `r == 0`.
        """
        r = r.astype(np.float64)
        x = (r - self.beam_r) * self.dx + self.bx * self.cos_enc
        y = (r - self.beam_r) * self.dy + self.bx * self.sin_enc
        z = (r - self.beam_r) * self.dz + self.bz
        xyz_lidar = np.stack([x, y, z], axis=-1) * 1e-3

        xyz_sensor = (
            np.einsum("...c,dc->...d", xyz_lidar, self.rotation)
            + self.translation)
        xyz_sensor[r == 0] = 0.0
        return xyz_sensor


class PointCloud(spec.Transform[types.OSDepth, types.PointCloud]):
    """Calculate point cloud from Ouster Lidar depth data.

    Args:
        min_range: minimum range to include in the point cloud; if `None`,
            all points are included.
    """

    def __init__(self, min_range: float | None = None) -> None:
        self.cache = ConfigCache()
        self.lut_cache: dict[int, _LidarXYZLut] = {}
        self.min_range = min_range

    def __call__(self, data: types.OSDepth) -> types.PointCloud:
        """Convert Ouster Lidar depth data to point cloud.

        Args:
            data: Ouster Lidar depth data with staggered measurements.

        Returns:
            Point cloud data.
        """
        _batch = data.rng.shape[:2]

        info = self.cache[data.intrinsics]
        mid = id(info)

        if mid not in self.lut_cache:
            self.lut_cache[mid] = _LidarXYZLut.from_intrinsics(info)
        lut = self.lut_cache[mid]

        xyz_all = lut.compute_xyz(
            data.rng.reshape(-1, *data.rng.shape[-2:])
        ).astype(np.float32)

        xyz = []
        for pc in xyz_all:
            if self.min_range is not None:
                valid = (np.linalg.norm(pc, axis=-1) >= self.min_range)
            else:
                valid = np.any(pc != 0, axis=-1)
            valid = valid & np.all(~np.isnan(pc), axis=-1)

            xyz.append(pc.reshape(-1, 3)[valid.reshape(-1)])

        padded = np.zeros(
            (len(xyz), max(x.shape[0] for x in xyz), 3), dtype=np.float32)
        length = np.array([x.shape[0] for x in xyz], dtype=np.int32)
        for i, x in enumerate(xyz):
            padded[i, :x.shape[0]] = x

        return types.PointCloud(
            xyz=padded.reshape(*_batch, -1, 3),
            length=length.reshape(_batch),
            timestamps=data.timestamps)
