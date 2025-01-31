"""Run range-doppler-azimuth-elevation FFT.

Inputs:
    - `radar/iq`

Outputs:
    - `_radar/cube`. WARNING: this will take 2.66X the disk space of `iq`!
"""

import math
from functools import partial

import jax
import numpy as np
from beartype.typing import cast
from einops import rearrange
from roverd import Dataset, sensors
from tqdm import tqdm

from roverp import doppler_range_azimuth_elevation


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out",
        help="Output dataset path; can be a clone of the first dataset.")
    p.add_argument("-b", "--batch", type=int, default=128, help="Batch size.")
    p.add_argument(
        "-s", "--scale", type=float, default=1e-3,
        help="Constant scale to multiply the data by.")


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

    nd, tx, rx, nr2 = radar.channels["iq"].shape
    if (rx != 4) or (tx != 3):
        print("This script requires 3x4 mode.")
        exit(1)

    # Create `_radar` virtual sensor
    radar = out.create("_radar", exist_ok=True)
    out = radar.create("cube", {
        "format": "raw", "type": "c8", "shape": [nd, 8, 2, nr2 // 2],
        "desc": "Doppler-azimuth-elevation-range 4D radar cube."})

    @jax.jit
    def _process(frame):
        return rearrange(
            jax.vmap(partial(
                doppler_range_azimuth_elevation, complex=True
            ))(frame) * args.scale,
            "b d r a e -> b d a e r")

    out.consume(np.array(_process(frame)) for frame in stream)
