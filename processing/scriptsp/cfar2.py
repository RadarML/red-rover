"""Run CFAR and AOA estimation.

The output data has the same shape as the input range-doppler images, and
assumes each range-doppler bin is a unique point. Different channels then
denote the properties of each bin.

Inputs:
    - `radar/*`

Outputs:
    - `_radar/cfar`: CFAR mask for the specified threshold.
    - `_radar/aoa`: quantized angle of arrival.
"""

import math
from queue import Queue

import jax
import numpy as np
from beartype.typing import cast
from jax import numpy as jnp
from roverd import Dataset, channels, sensors
from scipy.stats import norm
from tqdm import tqdm

from roverp import CFAR, AOAEstimation, RadarProcessing


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out",
        help="Output dataset path; can be a clone of the first dataset.")
    p.add_argument("-b", "--batch", type=int, default=64, help="Batch size.")
    p.add_argument(
        "--cfar_threshold", type=float, default=0.05,
        help="CFAR mask threshold as a p-value.")
    p.add_argument(
        "-w", "--artifact_window", type=int, default=2048,
        help="Zero doppler artifact calculation window.")


def cfar_pipeline(source, args):
    """Create and close on CFAR pipeline."""
    sample = None
    for sample in source.iq_stream(batch=args.artifact_window):
        break
    if sample is None:
        print("IQ stream seems to be empty.")
        exit(1)

    _, _, tx, _, _ = sample.shape
    antenna = None if tx == 2 else [0, 2]

    proc = RadarProcessing(
        jnp.array(sample), pad=[0, 0, 120], hanning=False, antenna=antenna)
    shape = proc.shape[:-1]
    cfar = CFAR(guard=(5, 5), window=(10, 10))
    cfar_threshold = norm.isf(args.cfar_threshold)
    aoa = AOAEstimation(bins=256, angle=False)

    @jax.jit
    def cfar_func(frame):
        frame_rda = proc(frame)
        frame_cfar = jax.vmap(cfar)(frame_rda) > cfar_threshold
        frame_aoa = jax.vmap(
            jax.vmap(jax.vmap(aoa)))(frame_rda).astype(jnp.int8)
        return frame_cfar, frame_aoa

    return cfar_func, shape


def _main(args):
    ds = Dataset(args.path)
    if args.out is None:
        out = ds
    else:
        out = Dataset(args.out)

    source = cast(sensors.RadarData, ds["radar"])
    cfar_func, shape = cfar_pipeline(source, args)

    cfar_out = Queue()
    aoa_out = Queue()

    radar = out.create("_radar", exist_ok=True)
    cast(channels.LzmaFrameChannel, radar.create("cfar", {
        "format": "lzmaf", "type": "bool", "shape": shape,
        "desc": f"CFAR point mask with p={args.cfar_threshold:.3f}."}
    )).consume(cfar_out, batch=0, thread=True)
    cast(channels.LzmaFrameChannel, radar.create("aoa", {
        "format": "lzmaf", "type": "i1", "shape": shape,
        "desc": "Azimuth angle of arrival, in (-128, 127)."}
    )).consume(aoa_out, batch=0, thread=True)

    stream = tqdm(
        source.iq_stream(batch=args.batch),
        total=math.ceil(len(ds["radar"]) / args.batch))
    for frame in stream:
        frame_cfar, frame_aoa = cfar_func(frame)
        cfar_out.put(np.array(frame_cfar))
        aoa_out.put(np.array(frame_aoa))
    cfar_out.put(None)
    aoa_out.put(None)
