"""Generate range-doppler-azimuth images."""

import os
import math
from tqdm import tqdm

import numpy as np
import jax
from jax import numpy as jnp
from rover import Dataset, RadarData, RadarProcessing

from beartype.typing import cast


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out", default=None,
        help="Output channel name; defaults depend on `--mode`.")
    p.add_argument("-b", "--batch", type=int, default=128, help="Batch size.")
    p.add_argument(
        "-w", "--artifact_window", type=int, default=2048,
        help="Zero doppler artifact calculation window; must fit in memory.")
    p.add_argument(
        "--mode", default="raw", help="FFT mode: raw, hann (with hanning "
        "window), hybrid (min(raw, hann)), rover1 (hybrid + nomask)")


DEFAULT_NAMES = {
    "hann": "hann", "raw": "raw", "hybrid": "rda", "rover1": "rover1"}


def _main(args):
    assert args.mode in DEFAULT_NAMES
    if args.out is None:
        args.out = DEFAULT_NAMES[args.mode]

    ds = Dataset(args.path)
    radar = cast(RadarData, ds["radar"])
    stream = tqdm(
        radar.iq_stream(batch=args.batch), desc=args.mode,
        total=math.ceil(len(ds["radar"]) / args.batch))

    if args.mode == "hybrid" or args.mode == "rover1":
        for sample in radar.iq_stream(batch=args.artifact_window):
            sample = jnp.array(sample)
            proc_hann = RadarProcessing(sample, hanning=True)
            proc_raw = RadarProcessing(sample, hanning=False)
            shape = proc_raw.shape
            break

        @jax.jit
        def _process(frame):
            return jnp.minimum(proc_hann(frame), proc_raw(frame))

    else:
        for sample in radar.iq_stream(batch=args.artifact_window):
            process = RadarProcessing(
                jnp.array(sample), hanning=(args.mode == "hann"))
            shape = process.shape
            break
        _process = jax.jit(process)

    # Create `_radar` virtual sensor
    radar = ds.create("_radar", exist_ok=True)
    rda_out = radar.create(args.out, {
        "format": "raw", "type": "f32", "shape": shape,
        "desc": "Doppler-range-azimuth image (mode={}).".format(args.mode)})
    ts_out = radar.create("ts", {
        "format": "raw", "type": "f64", "shape": [],
        "desc": "Smoothed timestamp, in seconds."})

    # Writeout
    ts = ds["radar"]["ts"].read()
    if args.mode == "rover1":
        ts_out.write(ts)
        rda_out.consume(_process(frame) for frame in stream)
    else:
        mask = np.load(os.path.join(args.path, "_radar", "pose.npz"))["mask"]
        ts_out.write(ts[mask])
        rda_out.consume(
            _process(frame)[mask[i * args.batch:(i + 1) * args.batch]]
            for i, frame in enumerate(stream))
