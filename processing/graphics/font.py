"""GPU-accelerated font rendering."""

from jax import numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from beartype.typing import Iterable, Union
from jaxtyping import Array, UInt8


class JaxFont:
    """GPU-accelerated vectorizable monospace text rendering.

    NOTE: `@cached_property` doesn't seem to play well with jax, so `JaxFont`
    pre-computes the font LUT. Don't intialize this class until needed!
    
    Parameters
    ----------
    font: Font file; must be monospace (or will be treated like one!)
    size: Font size; is static to allow pre-computing the font.
    """

    def __init__(self, font: str, size: int = 18) -> None:
        ttf = ImageFont.truetype(font, size, encoding="ascii")
        chars = bytes(list(range(32, 127))).decode('ascii')

        width, height = 0, 0
        for char in chars:
            _, _, w, h = ttf.getbbox(char)
            width = max(width, w)
            height = max(height, h)

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

        Parameters
        ----------
        text: character bytes (ASCII-encoded)
        canvas: array to write to. Must be a jax array.
        color: color to apply, with the same number of channels as `canvas`.
        x, y: position to write text at. Must be constant!

        Returns
        -------
        Rendered canvas. If the original is no longer used (e.g. all subsequent
            computation uses only the return here), `render` will not cause
            a copy as long as it is jit-compiled.
        """

        indices = jnp.clip(text - 32, 0, self.raster.shape[0] - 1)
        mask = jnp.concatenate(self.raster[indices], axis=1)
        b = y + mask.shape[0]
        r = x + mask.shape[1]
        mask = mask[
            :min(canvas.shape[0] - y, mask.shape[0]),
            :min(canvas.shape[1] - x, mask.shape[1])
        ] / 255
        
        return canvas.at[y:b, x:r].set(
            ((1 - mask)[:, :, None] * color[None, None, :]).astype(jnp.uint8)
            + (mask[:, :, None] * canvas[y: b, x: r]).astype(jnp.uint8))
    
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
