"""GPU-accelerated 2D graphics using JAX.

.. [G1] Matplotlib HSV to RGB implementation.
    https://matplotlib.org/3.1.1/_modules/matplotlib/colors.html#hsv_to_rgb
"""

from .colors import hsv_to_rgb, lut, mpl_colormap, render_image
from .font import JaxFont
from .render import Render
from .resize import resize
from .sync import synchronize
from .writer import write_buffered, write_consume

__all__ = [
    "hsv_to_rgb", "mpl_colormap", "lut", "render_image",
    "JaxFont", "Render",
    "resize", "synchronize",
    "write_buffered", "write_consume"]
