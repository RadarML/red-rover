"""2D rendering."""

import jax
import numpy as np
from beartype.typing import Any, Callable
from jax import numpy as jnp
from jaxtyping import Array, Shaped, UInt8

from .font import JaxFont


class Render:
    """2D renderer to combine data channels in a fixed format.

    Args:
        size: total frame size as `(height, width)`.
        channels: which data channels to render. Specify as
            `(xmin, xmax, ymin, ymax)`, where `x` is the vertical axis
            (lower is higher) and `y` is the horizontal axis (lower is left).
        transforms: dict of jax jit-compatible transforms to apply to each data
            channel (organized by channel name); must output RGB images.
        text: text to render; each key is a `(x, y)` coordinate, and each
            value is a format string. **NOTE**: if the strings ever change
            length, this will trigger recompilation!
        font, textcolor: text rendering configuration.
    """

    def __init__(
        self, size: tuple[int, int],
        channels: dict[tuple[int, int, int, int], str],
        transforms: dict[str, Callable[
            [Shaped[Array, "..."]], UInt8[Array, "?h ?w 3"]]],
        text: dict[tuple[int, int], str],
        font: JaxFont,
        textcolor: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.size = size
        self.font = font
        self.text = text

        def _get_transform(
            name: str
        ) -> Callable[[Shaped[Array, "..."]], UInt8[Array, "?h ?w 3"]]:
            tf = transforms.get(name)
            if tf is None:
                tf = transforms.get("*")
            if tf is None:
                tf = lambda x: x
            return tf

        def _render_func(
            data: dict[str, Shaped[Array, "..."]],
            encoded_text: dict[tuple[int, int], UInt8[Array, "..."]]
        ) -> UInt8[Array, "h w 3"]:
            frame = jnp.zeros((*size, 3), dtype=jnp.uint8)
            data = {k: _get_transform(k)(v) for k, v in data.items()}
            for k, v in channels.items():
                try:
                    frame = frame.at[k[0]:k[1], k[2]:k[3]].set(data[v])
                except ValueError as e:
                    print(
                        f"Incompatible shapes for channel {v}: "
                        f"{data[v].shape} into {k}")
                    raise e

            _textcolor = jnp.array(textcolor, dtype=jnp.uint8)
            for k2, v2 in encoded_text.items():
                frame = font.render(v2, frame, _textcolor, x=k2[0], y=k2[1])

            return frame

        self._render_func = jax.jit(_render_func)
        self._vrender_func = jax.jit(jax.vmap(_render_func))

    def render(
        self, data: dict[str, Shaped[Array, "..."]],
        meta: dict[str, Any] | list[dict[str, Any]]
    ) -> UInt8[Array, "*batch h w 3"]:
        """Render (possibly batched) frame.

        Args:
            data: input data, organized into channels by name. Must have
                fixed dimensions.
            meta: metadata for text captions/labels.

        Returns:
            Rendered RGB frame (or batch of frames).
        """
        # Batched
        if isinstance(meta, list):
            encoded_text = {
                k: jnp.stack([self.font.encode(v.format(**m)) for m in meta])
                for k, v in self.text.items()}
            return self._vrender_func(data, encoded_text)
        # Non-batched
        else:
            encoded_text = {
                k: self.font.encode(v.format(**meta))
                for k, v in self.text.items()}
            return self._render_func(data, encoded_text)
