"""Create ground truth occupancy grid from point cloud (.ply)."""

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
        "--padding", type=float, nargs='+', default=[5.0, 5.0, 2.5],
        help="Region padding relative to trajectory min/max.")
    p.add_argument(
        "--resolution", default=50.0, type=float,
        help="Grid resolution in grid cells per meter.")


def _set_bounds(args):
    args.padding = np.array((args.padding * 3)[:3])
    traj = RawTrajectory.from_csv(
        os.path.join(args.path, "_slam", "trajectory.csv"))
    lower = np.min(traj.xyz, axis=1) - args.padding
    upper = np.max(traj.xyz, axis=1) + args.padding
    return lower, upper


def _main(args):
    lower, upper = _set_bounds(args)

    data = PlyData.read(
        os.path.join(args.path, "_slam", "lidar.bag_points.ply"))
    x = data['vertex']['x']
    y = data['vertex']['y']
    z = data['vertex']['z']

    size = [
        math.ceil((u - lw) * args.resolution)
        for lw, u in zip(lower, upper)]
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
