"""Run CFAR and AOA estimation.

Inputs: `radar/*`
Outputs: `_cfar/*`
"""

import math
from tqdm import tqdm

import jax
from jax import numpy as jnp

from beartype.typing import cast
from jaxtyping import Float, Array

from rover import Dataset, RadarData, RadarProcessing, AOAEstimation, CFAR
from arrow import AWR1843Boost


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out",
        help="Output dataset path; can be a clone of the first dataset.")
    p.add_argument("-b", "--batch", type=int, default=64, help="Batch size.")
    p.add_argument(
        "-w", "--artifact_window", type=int, default=2048,
        help="Zero doppler artifact calculation window.")


def _main(args):
    ds = Dataset(args.path)
    if args.out is None:
        out = ds
    else:
        out = Dataset(args.out)

    radar = cast(RadarData, ds["radar"])
    stream = tqdm(
        radar.iq_stream(batch=args.batch),
        total=math.ceil(len(ds["radar"]) / args.batch))

    # Grab the first sample
    sample = None
    for sample in radar.iq_stream(batch=args.artifact_window):
        break
    if sample is None:
        print("IQ stream seems to be empty.")
        exit(1)

    _, _, tx, _, _ = sample.shape
    antenna = None if tx == 2 else [0, 2]

    proc = RadarProcessing(jnp.array(sample), hanning=False, antenna=antenna)
    shape = proc.shape[:-1]
    cfar = CFAR(guard=(5, 5), window=(10, 10))
    aoa = AOAEstimation(bins=1024)
    gain = AWR1843Boost(backend=jnp)

    @jax.jit
    def _cfar(frame):
        frame_rda: Float[Array, "batch doppler range antenna"] = proc(frame)
        frame_cfar = jax.vmap(cfar)(frame_rda)
        frame_aoa = jax.vmap(jax.vmap(jax.vmap(aoa)))(frame_rda)
        frame_gain = jnp.mean(frame_rda, axis=-1) * jax.vmap(
            jax.vmap(gain.gain_1x1)
        )(jnp.zeros_like(frame_aoa), frame_aoa)[..., 0]

        return {"amplitude": frame_gain, "cfar": frame_cfar, "aoa": frame_aoa}

    # Create `_cfar` virtual sensor
    radar = out.create("_cfar", exist_ok=True)

    _channels = {
        "amplitude": "CFAR point magnitude.",
        "cfar": "CFAR point inclusion threshold, in standard deviations.",
        "aoa": "Azimuth angle of arrival, in (-pi, pi)."}
    out = {
        k: radar.create(k, {
            "format": "raw", "type": "f4", "shape": shape, "desc": v})
        for k, v in _channels.items()}

    for frame in stream:
        processed = _cfar(frame)
        for k, v in out.items():
            v.write(processed[k], 'ab')
