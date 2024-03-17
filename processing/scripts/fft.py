"""Generate range-doppler-azimuth images."""

import os, json
import math
from functools import partial
from tqdm import tqdm

import numpy as np
import jax
from jax import numpy as jnp
from rover import Dataset, RadarData, dopper_range_azimuth

from beartype.typing import cast


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out", default='_radar',
        help="Output sensor name; defaults to `_radar`.")
    p.add_argument("-b", "--batch", type=int, default=128, help="Batch size.")
    p.add_argument(
        "--hanning", default=[0, 1], type=int, nargs='+',
        help="Axes to perform hanning window processing on "
        "(0:doppler, 1:range, 2:azimuth).")
    p.add_argument(
        "--nomask", default=False, action='store_true',
        help="Don't apply valid mask; no longer requires `_radar/pose.npz`.")


def _main(args):
    ds = Dataset(args.path)
    out_dir = os.path.join(args.path, args.out)
    stream = tqdm(
        cast(RadarData, ds["radar"]).iq_stream(batch=args.batch),
        desc="Initial FFT", total=math.ceil(len(ds["radar"]) / args.batch))

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

    # Create `_radar` virtual sensor
    radar = ds.create(args.out)
    rda_out = radar.create("rda", {
        "format": "raw", "type": "f32", "shape": rda[0].shape[1:],
        "desc": "Doppler-range-azimuth image (hann={}, nomask={}).".format(
            args.hanning, args.nomask)})
    ts_out = radar.create("ts", {
        "format": "raw", "type": "f64", "shape": [],
        "desc": "Smoothed timestamp, in seconds."})

    # Writeout
    ts = ds["radar"]["ts"].read()
    rda_pbar = tqdm(rda, desc="DC Correction & Writeout")
    if args.nomask:
        ts_out.write(ts)
        rda_out.consume(dc_correction(frame) for frame in rda_pbar)
    else:
        mask = np.load(os.path.join(args.path, "_radar", "pose.npz"))["mask"]
        ts_out.write(ts[mask])
        rda_out.consume(
            dc_correction(frame)[mask[i * args.batch:(i + 1) * args.batch]]
            for i, frame in enumerate(rda_pbar))
