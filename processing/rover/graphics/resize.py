"""GPU-accelerated nearest neighbor resizing."""


from jax import numpy as jnp

from jaxtyping import Array, Num


def resize(
    image: Num[Array, "h w ..."], height: int, width: int
) -> Num[Array, "h2 w2 ..."]:
    """Resize using nearest neighbor interpolation."""
    if image.shape[0] == height and image.shape[1] == width:
        return image

    ylut = jnp.rint(jnp.linspace(0, image.shape[0], height)).astype(jnp.uint16)
    xlut = jnp.rint(jnp.linspace(0, image.shape[1], width)).astype(jnp.uint16)
    return image[ylut, :][:, xlut]
