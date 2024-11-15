"""Pointcloud rendering utilities."""

import numpy as np
from beartype.typing import Sequence
from jax import numpy as jnp
from jaxtyping import Array, Integer, Num


def _offset_maximum(
    x: Num[Array, "*dims"], y: Num[Array, "*dims"], offsets: Sequence[int]
) -> Num[Array, "*dims"]:
    """Take the maximum between an image and an offset version of itself.

    Used for computing image dilations.
    """
    s1 = tuple([
        slice(max(0, off), n + min(off, 0))
        for n, off in zip(x.shape, offsets)])
    s2 = tuple([
        slice(max(0, -off), n + min(-off, 0))
        for n, off in zip(x.shape, offsets)])
    return x.at[s1].set(jnp.maximum(x[s1], y[s2]))


class Dilate:
    """Dilate an N-dimensional image.

    Takes the maximum of a ND image with offset version of itself.

    Args:
        radius: dilation radius; a neighborhood with size
            `floor(radius) * 2 + 1` is masked to points which are within
            `radius` of the center, and applied as the dilation mask.
        dims: number of dimensions.
    """

    def __init__(self, radius: float = 3.1, dims: int = 2) -> None:
        iradius = int(radius)
        window = [np.arange(-iradius, iradius + 1)] * dims
        coords = np.meshgrid(*window)
        self.mask: list[Integer[np.ndarray, "N"]] = [
            x - iradius
            for x in np.where(sum(x**2 for x in coords) < radius**2)]

    def __call__(self, image: Num[Array, "*dims"]) -> Num[Array, "*dims"]:
        """Apply dilation to an image."""
        dilated = image
        for coords in zip(*self.mask):
            dilated = _offset_maximum(dilated, image, coords)
        return dilated


class Scatter:
    """Render a scatter plot.

    Args:
        radius: point radius.
        resolution: image resolution.
    """

    def __init__(
        self, radius: float = 3.1,
        resolution: tuple[int, int] = (320, 640)
    ) -> None:
        self.dilate = Dilate(radius=radius, dims=2)
        self.resolution = resolution

    def __call__(
        self, x: Num[Array, "N"], y: Num[Array, "N"], c: Num[Array, "N"]
    ) -> Num[Array, "height width"]:
        """Render scatter plot image, with each point as a circle.

        Args:
            x: x-coordinate, with +x facing right. `x` should be
                normalized to [0, 1] as the width of the image.
            y: y-coordinate, with +y facing up. `y` should be normalized to
                [0, 1] as the height of the image.
            c: point intensity with arbitrary type. The intensity is sorted
                in increasing order to maximize the chances that the higher
                value is taken in case multiple points map to the same bin.

        Returns:
            Rendered scatter plot. Note that there is some indeterminism in
            case multiple points with the same intensity fall in the same
            initial pixel.
        """
        img = jnp.zeros(self.resolution, dtype=c.dtype)

        ord = jnp.argsort(c)
        iy = (self.resolution[0] * (1 - y)).astype(jnp.int32)
        ix = (x * self.resolution[1]).astype(jnp.int32)
        img = img.at[iy[ord], ix[ord]].set(c[ord])
        return self.dilate(img)
