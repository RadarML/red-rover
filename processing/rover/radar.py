"""Radar FFT processing.

NOTE: these routines are GPU-accelerated using JAX, which incurs significant
call and initialization overhead. As long as you are using jaxtyping > 0.2.26,
jax will not be imported until a function which requires it is called.
"""

from functools import partial
import numpy as np
import jax
from jax.scipy.signal import convolve2d
from jax import numpy as jnp

from jaxtyping import Float32, Array, Complex64, Bool
from beartype.typing import Union


def doppler_range_azimuth(
    iq: Complex64[Array, "doppler tx rx range"],
    hanning: Union[list[int], bool] = False, pad: int = 0
) -> Float32[Array, "doppler range antenna"]:
    """Compute doppler-range-azimuth FFTs with optional hanning window(s).

    Notes
    -----
    When applying a hanning window, we also normalize so that the total measure
    is preserved (e.g. fft(x) and fft(x * hann) have the same magnitude).
 
    Parameters
    ----------
    iq: input IQ array, in doppler-tx-rx-range order.
    hanning: list of indices to apply a Hanning window to; defaults to `[0, 2]`
        (i.e. range-doppler). Must be closed on in order to jit-compile.
    pad: apply zero-padding in the antenna axis.

    Notes
    -----
    Axes are indexed as follows:
    - 0: doppler
    - 1: range
    - 2: azimuth (antenna)
    """
    iqa: Complex64[Array, "doppler range antenna"] = jnp.swapaxes(
        iq.reshape(iq.shape[0], -1, iq.shape[-1]), -2, -1)

    if pad > 0:
        zeros = jnp.zeros((*iqa.shape[:2], pad), dtype=jnp.complex64)
        iqa = jnp.concatenate([iqa, zeros], axis=2)

    if isinstance(hanning, bool):
        hanning = [0, 1] if hanning else []
    for axis in hanning:
        shape = [1, 1, 1]
        shape[axis] = -1
        hann = jnp.hanning(iqa.shape[axis])
        window = hann.reshape(shape) / jnp.mean(hann)
        iqa = iqa * window

    rda = jnp.fft.fftn(iqa, axes=(0, 1, 2))
    rda_shf = jnp.fft.fftshift(rda, axes=(0, 2))
    return jnp.abs(rda_shf)


class RadarProcessing:
    """Doppler-range-azimuth FFT with zero-doppler artifact removal.
    
    Parameters
    ----------
    sample: sample IQ data for one-time artifact computation.
    hanning: whether to apply a hanning window in the range-doppler axes.
    """

    def __init__(
        self, sample: Complex64[Array, "batch doppler tx rx range"],
        hanning: bool = False
    ) -> None:
        self.hanning = hanning

        zero = sample.shape[1] // 2
        start = zero - (1 if hanning else 0)
        stop = zero + 1 + (1 if hanning else 0)
        self.patch = (slice(None), slice(start, stop))
        
        rda = jax.vmap(partial(
            doppler_range_azimuth, hanning=self.hanning))(sample)
        self.artifact = jnp.median(rda[self.patch], axis=0)[None, :, :, :]
        self.shape = rda.shape[1:]

    def __call__(
        self, iq: Complex64[Array, "batch doppler tx rx range"]
    ) -> Float32[Array, "batch doppler range antenna"]:
        """Run radar processing pipeline.
        
        Parameters
        ----------
        iq: batch of IQ data to run.

        Returns
        -------
        Doppler-range-antenna batch, with zero doppler correction applied.
        """
        rda = jax.vmap(partial(
            doppler_range_azimuth, hanning=self.hanning))(iq)
        corrected = rda.at[self.patch].set(
            jnp.maximum(rda[self.patch] - self.artifact, 0.0))
        return corrected


class CFAR:
    """Cell-averaging CFAR.
    
    Structured as a class to create averaging masks, etc outside of jax, then
    call again to use as a pure function.
    """

    def __init__(
        self, guard_band: tuple[int, int] = (2, 2),
        window_size: tuple[int, int] = (4, 4)
    ) -> None:
        w0, w1 = window_size
        g0, g1 = guard_band
        mask = np.ones((2 * w0 + 1, 2 * w1 + 1), dtype=np.float32)
        mask[w0 - g0: w0 + g0 + 1, w1 - g1: w1 + g1 + 1] = 0.0
        self.mask = jnp.array(mask)

    def __call__(self, rd: Float32[Array, "d r"]) -> Bool[Array, "d r"]:
        """Find CFAR values; the caller can then threshold as they see fit."""
        # Jax currently only supports 'fill', but this should be changed to
        # 'wrap' if they ever decide to add support.
        denom = convolve2d(jnp.ones_like(rd), self.mask, mode='same')
        cell_sum = convolve2d(rd, self.mask, mode='same')
        return rd / (cell_sum / denom)
