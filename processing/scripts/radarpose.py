"""Calculate interpolated poses for radar processing."""

import os
import json
import numpy as np

from rover import Dataset, Trajectory


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-s", "--smoothing", type=float, default=10.0,
        help="Smoothing coefficient; higher = more smooth.")


def _main(args):
    cfg = {
        "smoothing": args.smoothing,
        "start_threshold": 1.0,
        "filter_size": 5}

    traj = Trajectory(
        path=os.path.join(args.path, "_slam", "trajectory.csv"), **cfg)
    t_radar = Dataset(args.path)["radar"]["ts"].read()
    poses, mask = traj.interpolate(t_radar)

    np.savez(
        os.path.join(args.path, "_slam", "radar.npz"),
        **poses._asdict(), mask=mask)
    with open(os.path.join(args.path, "_slam", "slam.json"), 'w') as f:
        json.dump(cfg, f)
