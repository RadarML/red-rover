"""Convert to legacy DART format.

**NOTE**: this script is purely to provide compatibility for the original
rover/dart data format. This format is deprecated, and should not be used.

Inputs:
    - `radar/`
    - `_radar/pose.npz`
    - `_radar/rover1`

Outputs:
    - `_rover1/*`
"""

import json
import os

import numpy as np
from roverd import Dataset
from scipy.ndimage import binary_dilation


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out", default=None, help="Output directory; defaults to "
        "`_rover1/` in the dataset folder.")
    p.add_argument(
        "--min_speed", help="Minimum speed threshold (m/s).",
        default=0.2, type=float)
    p.add_argument(
        "--max_accel", help="Maximum allowed acceleration (m/s^2).",
        default=4.0, type=float)
    p.add_argument(
        "--accel_excl", help="Exclusion width for acceleration violations.",
        default=15, type=int)
    p.add_argument(
        "--speed_excl", help="Exclusion width for speed violations.",
        default=5, type=int)


def _main(args):
    import h5py

    if args.out is None:
        args.out = os.path.join(args.path, "_rover1")
    os.makedirs(args.out, exist_ok=True)

    poses = np.load(os.path.join(args.path, "_radar", "pose.npz"))

    speed_violation = binary_dilation(
        (np.linalg.norm(poses['vel'], ord=2, axis=1) < args.min_speed),
        np.ones(args.speed_excl, dtype=bool))
    accel_violation = binary_dilation(
        (np.linalg.norm(poses['acc'], ord=2, axis=1) > args.max_accel),
        np.ones(args.accel_excl, dtype=bool))
    valid = ~(speed_violation | accel_violation)  # type: ignore

    print("Speed violation (<{}m/s): {}".format(
        args.min_speed, np.sum(speed_violation)))
    print("Acceleration violation (>{}m/s^2): {}".format(
        args.max_accel, np.sum(accel_violation)))
    print("Total valid: {}/{}".format(np.sum(valid), valid.shape[0]))

    # Pose data
    outfile = h5py.File(os.path.join(args.out, "trajectory.h5"), 'w')
    for k, v in poses.items():
        outfile.create_dataset(k, data=v)
    outfile.create_dataset("valid", data=valid)

    # Pose metadata
    with open(os.path.join(args.out, "metadata.json"), 'w') as f:
        keys = ["min_speed", "max_accel", "accel_excl", "speed_excl"]
        meta = {k: getattr(args, k) for k in keys}
        json.dump({
            "n_frames": valid.shape[0],
            "total_time": poses['t'][-1] - poses['t'][0], **meta
        }, f, indent=4)

    # Radar data
    ds = Dataset(args.path)
    radarfile = h5py.File(os.path.join(args.out, "radar.h5"), 'w')
    radardata = np.swapaxes(ds['_radar']['rover1'].read(), 1, 2) * 1e-6
    radarfile.create_dataset("rad", data=radardata)

    # Radar metadata
    with open(os.path.join(args.path, "radar", "radar.json")) as f:
        _intrinsics = json.load(f)
        dr = _intrinsics["range_resolution"]
        dd = _intrinsics["doppler_resolution"]
        nd = _intrinsics["shape"][0]
        nr = _intrinsics["shape"][3]

    intrinsics = {
        "gain": "awr1843boost_az8",
        "r": [dr / 2, dr / 2 + (nr - 1) * dr, nr],
        "d": [-dd * nd / 2, dd * (nd - 1) / 2, nd]
    }
    with open(os.path.join(args.out, "sensor.json"), 'w') as f:
        json.dump(intrinsics, f, indent=4)
