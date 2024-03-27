"""Create ground truth occupancy grid.

Inputs: `_slam/trajectory.csv`, `_slam/lidar.bag_points.ply`
Outputs: `_rover1/map.npz` or `_slam/map.npz`, depending on `--legacy`.
"""

import os
import math
from tqdm import tqdm
import numpy as np
from plyfile import PlyData

from rover import RawTrajectory


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-l", "--legacy", default=False, action='store_true',
        help="Write in legacy Rover 1 format.")
    p.add_argument(
        "-b", "--batch", default=16 * 1024 * 1024, help="Batch size.")
    p.add_argument(
        "--margin_xy", type=float, default=5.0,
        help="Margin along the horizontal plane.")
    p.add_argument(
        "--margin_z", type=float, default=2.0,
        help="Margin along the vertical axis.")
    p.add_argument(
        "--resolution", default=50.0, type=float,
        help="Grid resolution in grid cells per meter.")
    p.add_argument(
        "--align", type=int, default=16,
        help="Each map axis is rounded up to the nearest multiple of `align`.")


def _set_bounds(args):
    margin = np.array([args.margin_xy, args.margin_xy, args.margin_z])
    traj = RawTrajectory.from_csv(
        os.path.join(args.path, "_slam", "trajectory.csv"))
    lower = np.min(traj.xyz, axis=1) - margin
    upper = np.max(traj.xyz, axis=1) + margin
    return lower, upper


def _main(args):
    lower, upper = _set_bounds(args)

    data = PlyData.read(
        os.path.join(args.path, "_slam", "lidar.bag_points.ply"))
    x = data['vertex']['x']
    y = data['vertex']['y']
    z = data['vertex']['z']

    size = [
        int(((x + args.align - 1) // args.align) * args.align)
        for x in (upper - lower) * args.resolution]
    grid = np.zeros(size, dtype=bool)
    print("Creating {}x{}x{} map @ {} cells/m".format(*size, args.resolution))
    print("Format: {}".format("rover1" if args.legacy else "red-rover"))

    for _ in tqdm(range(math.ceil(x.shape[0] / args.batch))):
        ix = ((x[:args.batch] - lower[0]) * args.resolution).astype(int)
        iy = ((y[:args.batch] - lower[1]) * args.resolution).astype(int)
        iz = ((z[:args.batch] - lower[2]) * args.resolution).astype(int)

        mask = (
            (ix > 0) & (ix < size[0])
            & (iy > 0) & (iy < size[1])
            & (iz > 0) & (iz < size[2]))

        grid[ix[mask], iy[mask], iz[mask]] = True
        x = x[args.batch:]
        y = y[args.batch:]
        z = z[args.batch:]

    if args.legacy:
        np.savez_compressed(
            os.path.join(args.path, "_rover1", "map.npz"),
            grid=grid, lower=np.array(lower), upper=np.array(upper))
    else:
        np.savez_compressed(
            os.path.join(args.path, "_slam", "map.npz"),
            grid=np.packbits(grid), shape=grid.shape,
            lower=np.array(lower), upper=np.array(upper))
