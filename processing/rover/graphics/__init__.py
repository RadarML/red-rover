"""GPU-accelerated 2D graphics using JAX.

.. [G1] Matplotlib HSV to RGB implementation.
    https://matplotlib.org/3.1.1/_modules/matplotlib/colors.html#hsv_to_rgb
"""

from .font import JaxFont
from .colors import hsv_to_rgb, lut
from .resize import resize

__all__ = ["JaxFont", "hsv_to_rgb", "lut", "resize"]
