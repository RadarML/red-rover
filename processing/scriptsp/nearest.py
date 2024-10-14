"""Run nearest-neighbor simulation.

Inputs:
    - `_radar/pose.npz`
    - `_radar/rda`

Outputs:
    - `_radar/sim_nearest` (same shape, properties as the reference channel).
"""

import os

import numpy as np
from beartype.typing import cast
from roverd import Dataset, channels


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-v", "--val", type=float, default=0.2,
        help="Proportion of validation frames.")


def _main(args):
    radar = Dataset(args.path)["_radar"]
    gt = cast(channels.RawChannel, radar["rda"]).memmap()

    nval = int(gt.shape[0] * args.val)
    poses = np.load(os.path.join(args.path, "_radar", "pose.npz"))
    posevec = np.concatenate([poses["pos"], poses["vel"]], axis=1)

    dist = np.linalg.norm(
        posevec[:-nval][:, None, :] - posevec[-nval:][None, :, :], axis=2)
    nearest = np.argmin(dist, axis=0)
    idxs = np.concatenate([np.arange(gt.shape[0] - nval), nearest])

    out = radar.create("sim_nearest", {
        "format": "raw", "type": "f32", "shape": gt.shape[1:],
        "desc": "Nearest neighbor doppler-range-azimuth image."})
    out.write(gt[idxs])
