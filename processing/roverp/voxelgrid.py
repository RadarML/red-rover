"""Voxel grid utilities."""

import matplotlib
import numpy as np
from beartype.typing import NamedTuple, Optional
from einops import einsum, rearrange
from jaxtyping import Array, Bool, Float, UInt8
from scipy.signal import convolve as default_conv


class VoxelGrid(NamedTuple):
    """Voxel grid of intensities.

    Attributes:
        data: voxel grid values.
        lower: coordinate of lower corner.
        resolution: resolution in units/m for each axis.
    """

    data: Float[np.ndarray, "Nx Ny Nz"]
    lower: Float[np.ndarray, "3"]
    resolution: Float[np.ndarray, "3"]

    @classmethod
    def from_npz(
        cls, path: str, key: str = "sigma", decimate: int = 1
    ) -> "VoxelGrid":
        """Load voxel grid, applying decimation factor if specified.

        Args:
            path: path to `.npz` file. Must have the specified `key`, as well
                as a `lower` and `resolution` array.
            key: field to load.
            decimate: decimation factor to apply.
        """
        npz = np.load(path)
        data = np.array(npz[key])

        if decimate > 1:
            data = einsum(
                rearrange(
                    data, "(Nx dy) (Ny dx) (Nz dz) -> Nx Ny Nz (dx dy dz)",
                    dx=decimate, dy=decimate, dz=decimate),
                "Nx Ny Nz d -> Nx Ny Nz")

        resolution = npz["resolution"] / decimate
        if len(resolution.shape) == 0:
            resolution = resolution.reshape(1)

        return cls(data=data, lower=npz["lower"], resolution=resolution)

    def crop(
        self, left: Optional[tuple[int, int, int]] = None,
        right: Optional[tuple[int, int, int]] = None
    ) -> "VoxelGrid":
        """Crop voxel grid by integer lower and upper indices.

        Args:
            left, right: lower and upper indices; `right` is interpreted as an
                ordinary index limit, so can be negative.
        """
        lower = self.lower
        data = self.data

        if left is not None:
            assert np.all(np.array(left) >= 0)
            data = data[left[0]:, left[1]:, left[2]:]
            lower = lower + np.array(left) / self.resolution
        if right is not None:
            data = data[:right[0], :right[1], :right[2]]

        return VoxelGrid(data=data, lower=lower, resolution=self.resolution)

    def cfar(
        self, guard_band: int = 1, window_size: int = 3,
        convolve_func=default_conv
    ) -> "VoxelGrid":
        """Get CFAR conv comparison values.

        Args:
            guard_band, window_size: CFAR window shape.
            convolve_func: convolution backend to use.
        """
        mask = np.ones([2 * window_size + 1] * 3)
        ax = slice(window_size - guard_band, window_size + guard_band + 1)
        mask[ax, ax, ax] = 0
        mask /= np.sum(mask)

        return VoxelGrid(
            data=convolve_func(self.data, mask, mode='same'), lower=self.lower,
            resolution=self.resolution)

    def normalize(
        self, left: float = 10.0, right: float = 99.9
    ) -> "VoxelGrid":
        """Normalize to (0, 1).

        Args:
            left, right: left and right percentiles to clip to.
        """
        ll, rr = np.percentile(self.data, [left, right])
        return VoxelGrid(
            data=np.clip((self.data - ll) / (rr - ll), 0.0, 1.0),
            lower=self.lower, resolution=self.resolution)

    def as_pointcloud(
        self, mask: Bool[Array, "Nx Ny Nz"], cmap: str = 'inferno'
    ) -> tuple[Float[np.ndarray, "3"], UInt8[np.ndarray, "3"]]:
        """Convert voxel grid to point cloud.

        Args:
            mask: mask of voxels to include.
            cmap: matplotlib colormap to use.

        Returns:
            (xyz, rgb), where xyz are positions and rgb colors.
        """
        colors = matplotlib.colormaps[cmap]

        ixyz = np.stack(np.where(mask))
        rgb = colors(self.data[ixyz[0], ixyz[1], ixyz[2]])[:, :3]
        rgb_u8 = (rgb * 255).astype(np.uint8)

        xyz = self.lower[:, None] + ixyz / self.resolution[:, None]

        return xyz, rgb_u8

    def as_pcd(
        self, path: str, mask: Bool[Array, "Nx Ny Nz"], cmap: str = 'inferno'
    ) -> None:
        """Save as a .pcd file.

        Args:
            path: output path (ending with .pcd).
            mask, cmap: pointcloud mask and colors.
        """
        from pypcd4 import PointCloud  # type: ignore

        xyz, rgb = self.as_pointcloud(mask, cmap=cmap)
        rgb_packed = PointCloud.encode_rgb(rgb)
        pc = PointCloud.from_xyzrgb_points([*xyz, rgb_packed])  # type: ignore
        pc.save(path)

    def as_ply(
        self, path: str, mask: Bool[Array, "Nx Ny Nz"], cmap: str = 'inferno'
    ) -> None:
        """Save as a .ply file.

        Args:
            path: output path (ending with .ply).
            mask, cmap: pointcloud mask and colors.
        """
        from plyfile import PlyData, PlyElement  # type: ignore

        xyz, rgb = self.as_pointcloud(mask, cmap=cmap)

        vertex = np.zeros(
            xyz.shape[1], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        vertex['x'] = xyz[0]
        vertex['y'] = xyz[1]
        vertex['z'] = xyz[2]
        vertex['red'] = rgb[:, 0]
        vertex['green'] = rgb[:, 1]
        vertex['blue'] = rgb[:, 2]
        el = PlyElement.describe(
            vertex, 'vertex', comments=['vertices with color'])
        PlyData([el], text=True).write(path)
