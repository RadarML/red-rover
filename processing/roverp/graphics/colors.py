"""JAX-native color conversions for GPU acceleration."""

import matplotlib
from beartype.typing import Optional, cast
from jax import numpy as jnp
from jaxtyping import Array, Float, Num, UInt8

from .resize import resize as resize_func


def hsv_to_rgb(
    hsv: Float[Array, "... 3"]
) -> Float[Array, "... 3"]:
    """Convert hsv values to rgb.

    Copied from [G1]_, modified for vectorization, and converted to jax.

    Args:
        hsv: HSV colors.

    Returns:
        RGB colors `float (0, 1)`, using the array format corresponding to the
        provided backend.
    """
    in_shape = hsv.shape
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = sum((i % 6 == j) * x for j, x in enumerate([v, q, p, p, t, v, v]))
    g = sum((i % 6 == j) * x for j, x in enumerate([t, v, v, q, p, p, v]))
    b = sum((i % 6 == j) * x for j, x in enumerate([p, p, t, v, v, q, v]))

    rgb = jnp.stack([r, g, b], axis=-1)
    return rgb.reshape(in_shape)


def lut(
    colors: Num[Array, "n d"], data: Float[Array, "..."]
) -> Num[Array, "... d"]:
    """Apply a discrete lookup table (e.g. colormap).

    Args:
        colors: list of discrete colors to apply (e.g. 0-255 RGB values). Can
            be an arbitrary number of channels, not just RGB.
        data: input data to index (`0 <= data <= 1`).

    Returns:
        An array with the same shape as `data`, with an extra dimension
        appended.
    """
    fidx = jnp.clip(data, 0.0, 1.0) * (colors.shape[0] - 1)
    return jnp.take(colors, fidx.astype(int), axis=0)


def mpl_colormap(cmap: str = "viridis") -> UInt8[Array, "n 3"]:
    """Get color LUT from matplotlib colormap.

    Use with :py:func:`graphics.lut`.
    """
    # For some reason, mypy does not recognize `colors` as an attribute of mpl.
    colors = cast(
        matplotlib.colors.ListedColormap,  # type: ignore
        matplotlib.colormaps[cmap]).colors
    return (jnp.array(colors) * 255).astype(jnp.uint8)


def render_image(
    data: Num[Array, "h w"],
    colors: Optional[Num[Array, "n d"]] = None,
    resize: Optional[tuple[int, int]] = None,
    scale: Optional[float | int] = None,
    pmin: Optional[float] = None, pmax: Optional[float] = None
) -> UInt8[Array, "h2 w2 d"]:
    """Apply colormap with specified scaling, clipping, and sizing.

    Args:
        colors: colormap (e.g. output of :py:func:`mpl_colormap`).
        data: input data to map.
        resize: resize inputs to specified size.
        scale: if specified, use this exact scale to normalize the data to
            `[0, 1]`, with clipping applied.
        pmin, pmax: if specified, use this percentile as the minimum and
            maximum for normalization instead of the actual min and max.

    Returns:
        Rendered RGB image.
    """
    if colors is None:
        raise ValueError("Must specify input colormap.")

    if scale is not None:
        data = jnp.clip(data / scale, 0.0, 1.0)
    else:
        left = (
            jnp.percentile(data, pmin) if pmin is not None else jnp.min(data))
        right = (
            jnp.percentile(data, pmax) if pmax is not None else jnp.max(data))
        data = jnp.clip((data - left) / (right - left), 0.0, 1.0)

    if resize is not None:
        data = resize_func(data, height=resize[0], width=resize[1])

    return lut(colors, data)
