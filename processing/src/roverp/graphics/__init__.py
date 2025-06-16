"""GPU-accelerated 2D graphics using JAX.

!!! warning

    This module is not automatically imported; you will need to explicitly
    import it:
    ```python
    from roverp import graphics
    ```
    You will also need to have the `graphics` extra installed.
"""

from .colors import hsv_to_rgb, lut, mpl_colormap, render_image
from .font import JaxFont
from .pointcloud import Dilate, Scatter
from .render import Render
from .resize import resize
from .sync import synchronize
from .writer import write_buffered, write_consume

__all__ = [
    "hsv_to_rgb", "mpl_colormap", "lut", "render_image",
    "Dilate", "Scatter",
    "JaxFont", "Render",
    "resize", "synchronize",
    "write_buffered", "write_consume"]
