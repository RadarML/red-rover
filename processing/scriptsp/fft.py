"""Run range-doppler-azimuth FFT.

Inputs:
    - `radar/*`
    - `_radar/pose.npz` (optional)

Outputs:
    - `_radar/{mode}` depending on the selected mode.

Channels:
    - `ts`: `_radar/ts` is populated if not already.
    - `{mode}`: see `--mode`.
"""

import os
import math
from tqdm import tqdm

import numpy as np
import jax
from jax import numpy as jnp
from beartype.typing import cast

from roverd import Dataset, sensors
from roverp import RadarProcessing


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out",
        help="Output dataset path; can be a clone of the first dataset.")
    p.add_argument("-b", "--batch", type=int, default=128, help="Batch size.")
    p.add_argument(
        "-w", "--artifact_window", type=int, default=2048,
        help="Zero doppler artifact calculation window.")
    p.add_argument(
        "--mode", default="raw", help="FFT mode: raw (no hann, nomask), "
        "hann (with hanning window, masked), hybrid (min(raw + mask, hann)), "
        "rover1 (hybrid + nomask)")
    p.add_argument(
        "--pad", default=[], nargs='+', type=int,
        help="range-doppler-azimuth-(elevation) padding.")


def _main(args):
    ds = Dataset(args.path)
    if args.out is None:
        out = ds
    else:
        out = Dataset(args.out)

    radar = cast(sensors.RadarData, ds["radar"])
    stream = tqdm(
        radar.iq_stream(batch=args.batch), desc=args.mode,
        total=math.ceil(len(ds["radar"]) / args.batch))

    # Grab the first sample
    sample = None
    for sample in radar.iq_stream(batch=args.artifact_window):
        break
    if sample is None:
        print("IQ stream seems to be empty.")
        exit(1)

    _, _, tx, rx, _ = sample.shape
    if (rx != 4) or (tx not in {2, 3}):
        print("This script only supports 2x4 or 3x4 mode.")
        exit(1)
    antenna = (None if tx == 2 else [0, 2])

    if args.mode == "hybrid" or args.mode == "rover1":
        sample = jnp.array(sample)
        proc_hann = RadarProcessing(
            sample, hanning=True, pad=args.pad, antenna=antenna)
        proc_raw = RadarProcessing(
            sample, hanning=False, pad=args.pad, antenna=antenna)
        shape = proc_raw.shape
        dtype = "f4"

        @jax.jit
        def _process(frame):
            return jnp.minimum(proc_hann(frame), proc_raw(frame))

    elif args.mode == "raw" or args.mode == "hann":
        process = RadarProcessing(
            jnp.array(sample), hanning=(args.mode == "hann"), antenna=antenna)
        shape = process.shape
        dtype = "f4"

        _process = jax.jit(process)

    else:
        print("Unknown mode: {}".format(args.mode))
        exit(1)

    # Create `_radar` virtual sensor
    radar = out.create("_radar", exist_ok=True)
    rda_out = radar.create(args.mode, {
        "format": "raw", "type": dtype, "shape": shape,
        "desc": "Doppler-range-azimuth image (mode={}).".format(args.mode)})

    # Writeout
    if args.mode in {"hann", "hybrid"}:
        ts = ds["radar"]["ts"].read()
        ts_out = radar.create("ts", {
            "format": "raw", "type": "f8", "shape": [],
            "desc": "Raw timestamp, in seconds (only for clipped modes)."})

        mask = np.load(os.path.join(args.path, "_radar", "pose.npz"))["mask"]
        ts_out.write(ts[mask])
        rda_out.consume(
            _process(frame)[mask[i * args.batch:(i + 1) * args.batch]]
            for i, frame in enumerate(stream))
    else:
        rda_out.consume(_process(frame) for frame in stream)
