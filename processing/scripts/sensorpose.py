"""Calculate interpolated poses for a specific sensor."""

import os
import numpy as np

from rover import Dataset, Trajectory


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-s", "--sensor", default="radar",
        help="Sensor timestamps to interpolate for.")
    p.add_argument(
        "-m", "--smoothing", type=float, default=10.0,
        help="Smoothing coefficient; higher = more smooth.")


def _main(args):
    cfg = {
        "smoothing": args.smoothing, "start_threshold": 1.0, "filter_size": 5}

    traj = Trajectory(
        path=os.path.join(args.path, "_slam", "trajectory.csv"), **cfg)
    t_radar = Dataset(args.path)[args.sensor]["ts"].read()
    poses, mask = traj.interpolate(t_radar)

    np.savez(
        os.path.join(args.path, "_" + args.sensor, "pose.npz"),
        **poses._asdict(), mask=mask, **cfg)
