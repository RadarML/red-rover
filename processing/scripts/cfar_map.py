"""Create reflectance grid from cfar data.

Inputs:
    - `_cfar/points.npz`
    - `_slam/trajectory.csv`

Outputs:
    - `_cfar/map.npz`.
    - Keys:
        - `grid`: reflectance grid with DART/Rover/Cartographer front-left-up
          axis conventions.
        - `lower`, `upper`: lower, upper corners of the grid.
"""

import os
import math
from tqdm import tqdm
import numpy as np

from rover import RawTrajectory


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "--margin_xy", type=float, default=5.0,
        help="Margin along the horizontal plane.")
    p.add_argument(
        "--margin_z", type=float, default=2.0,
        help="Margin along the vertical axis.")
    p.add_argument(
        "--resolution", default=25.0, type=float,
        help="Grid resolution in grid cells per meter.")
    p.add_argument(
        "--align", type=int, default=16,
        help="Each map axis is rounded up to the nearest multiple of `align`.")
    p.add_argument(
        "-b", "--batch", default=16,
        help="Batch size for approximate pointcloud maximum.")


def _main(args):
    lower, upper, size = RawTrajectory.from_csv(
        os.path.join(args.path, "_slam", "trajectory.csv")
    ).bounds(
        margin_xy=args.margin_xy, margin_z=args.margin_z,
        resolution=args.resolution, align=args.align)
    grid = np.zeros(size, dtype=np.float32)
    print("Creating {}x{}x{} map @ {} cells/m".format(*size, args.resolution))

    data = np.load(os.path.join(args.path, "_cfar", "points.npz"))
    sigma = data["amplitude"] / 1e6
    order = np.argsort(sigma)

    pos = data["pos"][order]
    sigma = sigma[order]

    for _ in tqdm(range(math.ceil(pos.shape[0] / args.batch))):

        x, y, z = ((pos[:args.batch] - lower) * args.resolution).astype(int).T
        mask = (
            (x > 0) & (x < size[0])
            & (y > 0) & (y < size[1])
            & (z > 0) & (z < size[2]))
        grid[x[mask], y[mask], z[mask]] = np.maximum(
            grid[x[mask], y[mask], z[mask]], sigma[:args.batch][mask])

        sigma = sigma[args.batch:]
        pos = pos[args.batch:]

    np.savez(
        os.path.join(args.path, "_cfar", "map.npz"),
        grid=grid, lower=lower, upper=upper)
