"""Generate range-doppler-azimuth images."""

import os
import json
import math
from functools import partial
from tqdm import tqdm

import jax
from jax import numpy as jnp
from rover import Dataset, RadarData, dopper_range_azimuth

from beartype.typing import cast


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out", default='_radar',
        help="Output sensor name; defaults to `_radar`.")
    p.add_argument(
        "-n", "--name", default="rda",
        help="Output channel name; defaults to `rda`.")
    p.add_argument(
        "--hanning", default=[0, 1], type=int, nargs='+',
        help="Axes to perform hanning window processing on.")


def _main(args):
    ds = Dataset(args.path)
    stream = tqdm(
        cast(RadarData, ds["radar"]).iq_stream(batch=128),
        desc="Initial FFT", total=math.ceil(len(ds["radar"]) / 128))

    # First step: convert to range-doppler. Record DC artifact (median).
    @jax.jit
    def to_rda(iq):
        rda_raw = jax.vmap(partial(
            dopper_range_azimuth, hanning=args.hanning))(iq)
        zero = rda_raw.shape[1] // 2
        dc = jnp.median(rda_raw[:, zero - 1:zero + 2], axis=0)
        return rda_raw, dc

    rda, _dc  = zip(*[to_rda(frame) for frame in stream])
    dc = jnp.median(jnp.stack(_dc), axis=0)

    # Second step: remove artifact to obtain corrected images.
    @jax.jit
    def dc_correction(rda):
        zero = rda.shape[2] // 2
        rda_corr = rda.at[:, zero - 1:zero + 2].set(
            jnp.maximum(rda[:, zero - 1:zero + 2] - dc[None, :, :], 0))

        return rda_corr

    # Writeout
    os.makedirs(os.path.join(args.path, args.out), exist_ok=True)
    with open(os.path.join(args.path, args.out, args.name), 'wb') as f:
        for frame in tqdm(rda, desc="DC Correction & Writeout"):
            f.write(dc_correction(frame).tobytes())

    # Metadata (append or write)
    meta = ds.get_metadata(args.out)
    meta[args.name] = {
        "format": "raw", "type": "f32", "shape": rda[0].shape[1:],
        "desc": "Doppler-range-azimuth image (hann={}).".format(args.hanning)
    }
    ds.write_metadata(args.out, meta)
