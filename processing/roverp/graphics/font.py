"""GPU-accelerated font rendering."""

import os

import numpy as np
from beartype.typing import Iterable, Optional, Union
from jax import numpy as jnp
from jaxtyping import Array, UInt8
from PIL import Image, ImageDraw, ImageFont


class JaxFont:
    """GPU-accelerated vectorizable monospace text rendering.

    NOTE: `@cached_property` doesn't seem to play well with jax, so `JaxFont`
    pre-computes the font LUT. Don't intialize this class until needed!

    Args:
        font: Font file; must be monospace (or will be treated like one!). If
            `None`, load the included `roboto.ttf` file.
        size: Font size; is static to allow pre-computing the font.

    Usage:
        1. Initialize: `font = JaxFont(font_name, size)`.
        2. Convert text to array(s): `arr = font.encode("Hello World!")`.
        3. Render onto canvas: `canvas = font.render(arr, canvas, color, x, y)`.
        4. Wrap any `render` calls into a JIT-compiled function to guarantee
           in-place editing.
    """

    def __init__(self, font: Optional[str] = None, size: int = 18) -> None:
        if font is None:
            font = os.path.join(os.path.dirname(__file__), "roboto.ttf")

        ttf = ImageFont.truetype(font, size, encoding="ascii")
        chars = bytes(list(range(32, 127))).decode('ascii')

        width, height = 0, 0
        for char in chars:
            _, _, w, h = ttf.getbbox(char)
            width = max(width, int(w))
            height = max(height, int(h))

        stack = []
        for char in chars:
            canvas = Image.new('L', (width, height), "white")
            ImageDraw.Draw(canvas).text((0, 0), char, 'black', ttf)
            stack.append(np.array(canvas))
        self.raster = jnp.stack(stack)

    def render(
        self, text: UInt8[Array, "len"],
        canvas: UInt8[Array, "width height channels"],
        color: UInt8[Array, "channels"], x: int = 0, y: int = 0
    ) -> UInt8[Array, "width height channels"]:
        """Render text on canvas.

        Args:
            text: character bytes (ASCII-encoded)
            canvas: array to write to. Must be a jax array.
            color: color to apply, with the same number of channels as `canvas`.
            x, y: position to write text at. Must be constant! **NOTE**:
                following pixel indexing convention, +x is down, and +y is
                right.

        Returns:
            Rendered canvas. If the original is no longer used (e.g. all
            subsequent computation uses only the return here), `render` will
            not cause a copy as long as it is jit-compiled.
        """
        indices = jnp.clip(text - 32, 0, self.raster.shape[0] - 1)
        mask = jnp.concatenate(self.raster[indices], axis=1)
        b = x + mask.shape[0]
        r = y + mask.shape[1]
        mask = mask[
            :min(canvas.shape[0] - x, mask.shape[0]),
            :min(canvas.shape[1] - y, mask.shape[1])
        ] / 255

        return canvas.at[x:b, y:r].set(
            ((1 - mask)[:, :, None] * color[None, None, :]).astype(jnp.uint8)
            + (mask[:, :, None] * canvas[x: b, y: r]).astype(jnp.uint8))

    def encode(
        self, text: Union[str, Iterable[str]]
    ) -> UInt8[Array, "..."]:
        """Convert a string or list of strings to an array of ASCII indices.

        NOTE: the inputs must all have the same length. This function is not
        jit-compilable.
        """
        if isinstance(text, str):
            return jnp.frombuffer(
                bytes(text, encoding='ascii'), dtype=np.uint8)
        else:
            return jnp.stack([
                np.frombuffer(bytes(s, encoding='ascii'), dtype=np.uint8)
                for s in text])
