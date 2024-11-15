"""Radar FFT processing.

NOTE: these routines are GPU-accelerated using JAX, which incurs significant
call and initialization overhead. As long as you are using jaxtyping > 0.2.26,
jax will not be imported until a function which requires it is called.
"""

from functools import partial

import jax
import numpy as np
from beartype.typing import Optional, Sequence, Union
from einops import rearrange
from jax import numpy as jnp
from jax.scipy.signal import convolve2d
from jaxtyping import Array, Complex64, Float, Float32, Int
from scipy.stats import norm


def doppler_range_azimuth(
    iq: Complex64[Array, "d tx rx rng"],
    hanning: Union[list[int], bool] = False, pad: list[int] = [],
    complex: bool = False
) -> Float32[Array, "d rng antenna"] | Complex64[Array, "d rng antenna"]:
    """Compute doppler-range-azimuth FFTs with optional hanning window(s).

    - The (virtual) antenna must be arranged in a line on the same elevation.
    - When applying a hanning window, we also normalize so that the total
      measure is preserved (e.g. `fft(x)` and `fft(x * hann)` have the same
      magnitude).

    Axes are indexed as follows:

    - 0: doppler
    - 1: range
    - 2: azimuth (antenna)

    Args:
        iq: input IQ array, in doppler-tx-rx-range order.
        hanning: list of indices to apply a Hanning window to; defaults to
            `[0, 2]` (i.e. range-doppler). Must be closed on in order to
            jit-compile.
        pad: padding to apply in the (doppler, range, azimuth) axes.
        complex: whether to return a complex number, or take the magnitude.

    Returns:
        (doppler, range, antenna) post-FFT radar data cube, with the
        appropriate fftshift.
    """
    iqa: Complex64[Array, "doppler range antenna"] = jnp.swapaxes(
        iq.reshape(iq.shape[0], -1, iq.shape[-1]), -2, -1)

    if isinstance(hanning, bool):
        hanning = [0, 1] if hanning else []
    for axis in hanning:
        shape = [1, 1, 1]
        shape[axis] = -1
        hann = jnp.hanning(iqa.shape[axis])
        window = hann.reshape(shape) / jnp.mean(hann)
        iqa = iqa * window

    for i, size in enumerate(pad):
        if size > 0:
            shape = list(iqa.shape)
            shape[i] = size
            zeros = jnp.zeros(shape, dtype=jnp.complex64)
            iqa = jnp.concatenate([iqa, zeros], axis=i)

    rda = jnp.fft.fftn(iqa, axes=(0, 1, 2))
    rda_shf = jnp.fft.fftshift(rda, axes=(0, 2))
    return rda_shf if complex else jnp.abs(rda_shf)


def doppler_range_azimuth_elevation(
    iq: Complex64[Array, "doppler 3 4 range"],
    raw_elevation: bool = False, hanning: Union[list[int], bool] = False,
    pad: list[int] = [], complex: bool = False
) -> (
    Float32[Array, "doppler range azimuth elevation"]
    | Complex64[Array, "doppler range azimuth elevation"]
):
    """Compute doppler-range-azimuth-elevation FFTs for the 3x4 TI AWR1843.

    See `doppler_range_azimuth` for additional documentation.

    Args:
        iq: input IQ array, in doppler-tx-rx-range order.
        hanning: list of indices to apply a Hanning window to.
        pad: padding to apply in the (doppler, range, azimuth, elevation) axes.
        complex: whether to return a complex number, or take the magnitude.

    Returns:
        (doppler, range, azimuth, elevation) radar data cube.
    """
    iq = rearrange(iq, "d tx rx r -> d tx r rx")
    iqa: Complex64[Array, "doppler range 8 2"] = jnp.zeros(
        (iq.shape[0], iq.shape[2], 8, 2), dtype=jnp.complex64
    ).at[:, :, 2:6, 0].set(iq[:, 1, :, :]
    ).at[:, :, 0:4, 1].set(iq[:, 0, :, :]
    ).at[:, :, 4:8, 1].set(iq[:, 2, :, :])

    if isinstance(hanning, bool):
        hanning = [0, 1] if hanning else []
    for axis in hanning:
        shape = [1, 1, 1, 1]
        shape[axis] = -1
        hann = jnp.hanning(iqa.shape[axis])
        window = hann.reshape(shape) / jnp.mean(hann)
        iqa = iqa * window

    for i, size in enumerate(pad):
        if size > 0:
            shape = list(iqa.shape)
            shape[i] = size
            zeros = jnp.zeros(shape, dtype=jnp.complex64)
            iqa = jnp.concatenate([iqa, zeros], axis=i)

    # Jax only allows up to 3D FFTs, so we need to do 2 FFTs here.
    rda = jnp.fft.fftn(iqa, axes=(0, 1, 2))
    rda = jnp.fft.fftn(rda, axes=(3,))
    rda = jnp.fft.fftshift(rda, axes=(0, 2, 3))
    return rda if complex else jnp.abs(rda)


def elevation_aoa(
    iq: Complex64[Array, "doppler 3 4 range"]
) -> Float[Array, "doppler range"]:
    """Calculate elevation  AOA and range-doppler intensity.

    Args:
        iq: input data. Must be in 3x4 mode. The median across the 4 central
            RX antenna is returned.

    Returns:
        Elevation AoA for the input dat.
    """
    iq_az: Complex64[Array, "doppler range 4 2"] = jnp.zeros(
        (iq.shape[0], iq.shape[-1], 4, 2), dtype=jnp.complex64
    ).at[:, :, :, 0].set(rearrange(iq[:, 1, :, :], "d rx r -> d r rx")
    ).at[:, :, 0:2, 1].set(rearrange(iq[:, 0, 2:, :], "d rx r -> d r rx")
    ).at[:, :, 2:4, 1].set(rearrange(iq[:, 2, :2, :], "d rx r -> d r rx"))

    angle = jnp.angle(iq_az)
    phase_diff = jnp.median(angle[..., 0] - angle[..., 1], axis=-1)
    el_angle = jnp.arcsin((phase_diff / jnp.pi + 1) % 2 - 1)

    return el_angle


class RadarProcessing:
    """Doppler-range-azimuth FFT with zero-doppler calibration.

    `RadarProcessing` performs the following steps:

    0. Calculate the zero doppler offset.

        - This is a constant offset applied to the zero-doppler bins of
          range-Doppler images caused by the antenna geometry and any returns
          from the data collection rig (which is mounted rigidly to the radar,
          and therefore has no return).
        - We assume that range-doppler plots are sparse, and take the median
          across time for the zero doppler bin to estimate this offset.
        - If a hanning window is applied, we instead calculate the offset
          across doppler bins [-1, 1] to account for doppler bleed.

    1. Take doppler-range-azimuth FFT.

        - If `hanning` is specified, we also apply a hanning window to the
          range and doppler axes.

    2. Apply the zero-doppler offset.

        - This offset only results in modifications to the zero doppler bin
          (and +/-1 bin if using a hanning window).
        - Any resulting values are clipped to be non-negative.

    Args:
        sample: sample IQ data for one-time artifact computation.
        hanning: whether to apply a hanning window in the range-doppler axes.
        pad: (doppler, range, azimuth) padding.
        antenna: only use a subset of TX antenna if specified.
    """

    def __init__(
        self, sample: Complex64[Array, "batch doppler tx rx range"],
        hanning: bool = False, pad: list[int] = [],
        antenna: Optional[list[int]] = None
    ) -> None:
        self.hanning = hanning
        self.pad = pad
        self.antenna = antenna

        if self.antenna is not None:
            sample = sample[:, :, self.antenna, :, :]

        # Not batched to make sure this always succeeds.
        @jax.jit
        def _get_artifact(frame):
            return doppler_range_azimuth(
                frame, hanning=self.hanning, pad=pad)[self.patch[1:]]

        self.shape = doppler_range_azimuth(
            sample[0], hanning=self.hanning, pad=pad).shape
        zero = self.shape[0] // 2
        start, stop = zero, zero

        if hanning:
            start -= 1
            stop += 1
        if len(pad) > 0:
            start -= 4
            stop += 4

        self.patch = (slice(None), slice(start, stop))
        self.artifact = jnp.median(jnp.array(
            [_get_artifact(x) for x in sample]), axis=0)[None, :, :, :]

    def __call__(
        self, iq: Complex64[Array, "batch doppler tx rx range"]
    ) -> Float32[Array, "batch doppler range antenna"]:
        """Run radar processing pipeline.

        Args:
            iq: batch of IQ data to run.

        Returns:
            Doppler-range-antenna batch, with zero doppler correction applied.
        """
        if self.antenna is not None:
            iq = iq[:, :, self.antenna, :, :]

        rda = jax.vmap(partial(
            doppler_range_azimuth, hanning=self.hanning, pad=self.pad))(iq)
        corrected = rda.at[self.patch].set(
            jnp.maximum(rda[self.patch] - self.artifact, 0.0))
        return corrected


class CFAR:
    """Cell-averaging CFAR.

    Expects a 2d input, with the `guard` and `window` sizes corresponding to
    the respective input axes.

    **Note**: The user is responsible for applying the desired thresholding.
    For example, when using a gaussian model, the threshold should be
    calculated using an inverse normal CDF (e.g. `scipy.stats.norm.isf`)::

        cfar = CFAR(guard=(2, 2), window=(4, 4))
        thresholds = cfar(image)
        mask = (thresholds > scipy.stats.norm.isf(0.01))

    Args:
        guard: size of guard cells (excluded from noise estimation).
        window: CFAR window size.
    """

    def __init__(
        self, guard: tuple[int, int] = (2, 2),
        window: tuple[int, int] = (4, 4)
    ) -> None:
        w0, w1 = window
        g0, g1 = guard

        mask = np.ones((2 * w0 + 1, 2 * w1 + 1), dtype=np.float32)
        mask[w0 - g0: w0 + g0 + 1, w1 - g1: w1 + g1 + 1] = 0.0
        self.mask = jnp.array(mask)

    def __call__(self, x: Float[Array, "d r ..."]) -> Float[Array, "d r"]:
        """Get CFAR thresholds.

        Args:
            x: input. If more than 2 axes are present, the additional axes
                are averaged before running CFAR.

        Returns:
            CFAR threshold values for this input.
        """
        # Collapse additional axes if required
        while len(x.shape) > 2:
            x = jnp.mean(x, axis=-1)

        # Jax currently only supports 'fill', but this should be changed to
        # 'wrap' if they ever decide to add support.
        valid = convolve2d(jnp.ones_like(x), self.mask, mode='same')
        mu = convolve2d(x, self.mask, mode='same') / valid
        second_moment = convolve2d(x**2, self.mask, mode='same') / valid
        sigma = jnp.sqrt(second_moment - mu**2)

        return (x - mu) / sigma


class AOAEstimation:
    """Angle of arrival estimation.

    Args:
        bins: number of angular bins to span `(-pi, pi)` during AOA estimation.
        angle: whether to return the angle or bin index.
    """

    def __init__(self, bins: int = 128, angle: bool = True) -> None:
        self.bins = bins
        self.angle = angle

    def __call__(
        self, x: Float[Array, "a"]
    ) -> Float[Array, ""] | Int[Array, ""]:
        """Estimate angle of arrival for a planar antenna array.

        Args:
            x: planar array receive values. Should already have any relevant
                FFTs applied.

        Returns:
            Estimated AOA in `(-pi, pi)` or bin in `(-bins / 2, bins / 2 - 1)`.
        """
        assert self.bins % x.shape[0] == 0

        n = jnp.arange(x.shape[0])
        bin = (jnp.arange(self.bins) / self.bins * 2 - 1) * np.pi
        lobes = jnp.abs(jnp.sum(
            jnp.exp(-1j * n[:, None] * bin[None, :]), axis=0))

        upsampled = jnp.zeros(self.bins).at[::self.bins // x.shape[0]].set(x)
        # FFT-based cirular convolution
        aoavec = jnp.abs(jnp.fft.fftshift(
            jnp.fft.ifft(jnp.fft.fft(upsampled) * jnp.fft.fft(lobes))))

        if self.angle:
            return (jnp.argmax(aoavec) / self.bins - 0.5) * np.pi
        else:
            return jnp.argmax(aoavec) - self.bins // 2


class CFARProcessing:
    """CFAR + 3D AoA Processing Pipeline.

    Args:
        sample: sample data for zero-doppler artifact correction.
        guard: guard band size for CFAR.
        window: test window size for CFAR.
        bins: azimuth Angle of Arrival estimation resolution (-pi/2 to pi/2).
        cfar_threshold: CFAR p-value threshold.
        n_points: number of CFAR points to return. If more CFAR points are
            detected, the top `n_points` are returned; if fewer, the points
            are zero-padded.
    """

    def __init__(
        self, sample: Complex64[Array, "batch doppler tx rx range"],
        guard: tuple[int, int] = (5, 5), window: tuple[int, int] = (10, 10),
        bins: int = 256, cfar_threshold: float = 0.005, max_points: int = 256
    ) -> None:
        self.cfar_threshold = cfar_threshold
        self.max_points = max_points

        self.proc = RadarProcessing(
            sample, hanning=True, antenna=[0, 2], pad=[0, 0, bins - 8, 0])
        self.cfar = CFAR(guard=guard, window=window)
        self.aoa = AOAEstimation(bins=bins, angle=True)

    def __call__(
        self, iq: Complex64[Array, "batch doppler tx rx range"]
    ) -> tuple[
        Float[Array, "N"], Float[Array, "N"],
        Float[Array, "N"], Float[Array, "N"]
    ]:
        """Apply CFAR-AoA pipeline to obtain polar points.

        Args:
            iq: 3x4 IQ data.

        Returns:
            A tuple with the `rng` (range), `az` (azimuth angle),
            `el` (elevation angle), and `intensity`. Exactly `max_points`
            CFAR points are returned.
        """
        rda = self.proc(iq)
        cfar = jax.vmap(self.cfar)(rda) > norm.isf(self.cfar_threshold)
        az_angle = jax.vmap(jax.vmap(jax.vmap(self.aoa)))(rda)
        el_angle = jax.vmap(elevation_aoa)(iq)
        intensity = cfar * jnp.linalg.norm(rda, axis=-1)

        def top_n(az, el, x):# -> tuple[Array, Any, Any, Any]:
            Nd, Nr = x.shape
            i_doppler, i_range = jnp.meshgrid(jnp.arange(Nd), jnp.arange(Nr))

            keep = jnp.argsort(
                x[i_doppler.reshape(-1), i_range.reshape(-1)], descending=True
            )[:self.max_points][::-1]
            d = i_doppler.reshape(-1)[keep]
            r = i_range.reshape(-1)[keep]
            return r, az[d, r], el[d, r], x[d, r]

        return jax.vmap(top_n)(az_angle, el_angle, intensity)
