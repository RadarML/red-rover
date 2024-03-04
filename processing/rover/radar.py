"""Radar FFT processing.

NOTE: these routines are GPU-accelerated using JAX, which incurs significant
call and initialization overhead. As long as you are using jaxtyping > 0.2.26,
jax will not be imported until a function which requires it is called.
"""

import numpy as np
from jax.scipy.signal import convolve2d
from jax import numpy as jnp

from jaxtyping import Float32, Array, Complex64, Bool


def range_doppler_azimuth(
    iq: Complex64[Array, "doppler tx rx range"],
    hanning: list[int] = [0, 2]
) -> Float32[Array, "range doppler antenna"]:
    """Compute range-doppler-azimuth FFTs.

    Parameters
    ----------
    iq: input IQ array, in doppler-tx-rx-range order.
    hanning: list of indices to apply a Hanning window to; defaults to `[0, 2]`
        (i.e. range-doppler). Must be closed on in order to jit-compile.

    Notes
    -----
    Axes are indexed as follows:
    - 0: doppler
    - 1: azimuth (antenna)
    - 2: range
    """
    iqa = iq.reshape(iq.shape[0], -1, iq.shape[-1])

    for axis in hanning:
        shape = [1, 1, 1]
        shape[axis] = -1
        window = jnp.hanning(iqa.shape[axis]).reshape(shape)
        iqa = iqa * window

    rda = jnp.fft.fftn(iqa, axes=(0, 1, 2))
    rda_shf = jnp.fft.fftshift(rda, axes=(0, 1))
    rda_real = jnp.moveaxis(jnp.abs(rda_shf), [0, 1, 2], [1, 2, 0])
    return rda_real


class CFAR:
    """Cell-averaging CFAR.
    
    Structured as a class to create averaging masks, etc outside of jax, then
    call again to use as a pure function.
    """

    def __init__(
        self, guard_band: tuple[int, int] = (2, 2),
        window_size: tuple[int, int] = (4, 4),
        threshold: float = 10.0, percentile: float = 0.0
    ) -> None:
        self.threshold = threshold
        self.percentile = percentile

        w0, w1 = window_size
        g0, g1 = guard_band
        mask = np.ones((2 * w0 + 1, 2 * w1 + 1), dtype=np.float32)
        mask[w0 - g0: w0 + g0 + 1, w1 - g1: w1 + g1 + 1] = 0.0
        self.mask = jnp.array(mask)

    def __call__(
        self, rd: Float32[Array, "range doppler"]
    ) -> Bool[Array, "range doppler"]:
        """Find CFAR mask."""
        # Jax currently only supports 'fill', but this should be changed to
        # 'wrap' if they ever decide to add support.
        denom = convolve2d(jnp.ones_like(rd), self.mask, mode='same')
        cell_sum = convolve2d(rd, self.mask, mode='same')
        ca_mask = rd > (cell_sum / denom * self.threshold)

        if self.percentile > 0:
            minval = jnp.percentile(rd, self.percentile)
            ca_mask = jnp.logical_and(ca_mask, rd > minval)

        return ca_mask
