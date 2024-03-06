"""GPU-accelerated nearest neighbor resizing."""


from jax import numpy as jnp

from jaxtyping import Array, Num


def resize(
    image: Num[Array, "w h ..."], height: int, width: int
) -> Num[Array, "w2 h2 ..."]:
    """Resize using nearest neighbor interpolation."""
    ylut = jnp.rint(jnp.linspace(0, image.shape[0], height)).astype(jnp.uint16)
    xlut = jnp.rint(jnp.linspace(0, image.shape[1], width)).astype(jnp.uint16)
    return image[ylut, :][:, xlut]
