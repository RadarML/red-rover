"""Create CFAR point cloud.

Inputs: `_cfar/*`, `_radar/pose.npz`
Outputs: `_cfar/pointcloud.npz`
"""

import numpy as np
import os, json
from tqdm import tqdm

from rover import Dataset


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument("-c", "--cfar", default=5.0, help="CFAR threshold floor.")
    p.add_argument(
        "-a", "--abs", default=1e5, help="Magnitude absolute floor.")


def _main(args):
    with open(os.path.join(args.path, "radar", "radar.json")) as f:
        intrinsics = json.load(f)

    def to_points(threshold, amplitude, aoa, pos, rot):
        _range, _doppler = np.meshgrid(
            np.arange(threshold.shape[0]), np.arange(threshold.shape[1]))

        # Reject if under CFAR/magnitude threshold, and also zero doppler
        mask = (
            (threshold > args.cfar)
            & (amplitude > args.abs)
            & (_doppler != threshold.shape[1] // 2)
        ).reshape(-1)

        pts_rng = _range.reshape(-1)[mask] * intrinsics['range_resolution']
        pts_aoa = aoa.reshape(-1)[mask]
        pts_amp = amplitude.reshape(-1)[mask]
        pts_thresh = threshold.reshape(-1)[mask]

        # Coordinate convention: +x is straight forward, +y is left
        # AOA measures clockwise relative to +x
        xyz_sensor = np.stack([
            np.cos(pts_aoa) * pts_rng,
            np.sin(-pts_aoa) * pts_rng,
            np.zeros_like(pts_aoa)])
        xyz_world = np.matmul(rot, xyz_sensor) + pos[:, None]

        return {"pos": xyz_world.T, "amplitude": pts_amp, "cfar": pts_thresh}


    cfar = Dataset(args.path).get("_cfar")
    poses = np.load(os.path.join(args.path, "_radar", "pose.npz"))

    # Toss tqdm in here so it fetches the progress bar length for us
    frames = zip(
        tqdm(cfar["cfar"].read()[poses["mask"]]),
        cfar["amplitude"].read()[poses["mask"]],
        cfar["aoa"].read()[poses["mask"]],
        poses["pos"],
        poses["rot"])

    out = {"pos": [], "amplitude": [], "cfar": []}
    for (thr, amp, aoa, pos, rot) in frames:
        pts = to_points(thr, amp, aoa, pos, rot)
        for k, v in pts.items():
            out[k].append(v)

    np.savez(
        os.path.join(args.path, "_cfar", "points.npz"),
        **{k: np.concatenate(v) for k, v in out.items()})
