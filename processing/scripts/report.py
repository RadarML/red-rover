"""Get speed report.

Inputs:
    - `_radar/pose.npz`
    - `_slam/trajectory.csv`

Outputs:
    - `_report/speed.pdf`
"""

import os
import math
import json
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import rover


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out",
        help="Output path; defaults to `_reports/...` in the dataset.")
    p.add_argument(
        "-w", "--width", default=30.0, type=float,
        help="Time series plot row width, in seconds.")


def _plot_speed(axs, tproc, traw, vproc, vraw, dmax):
    """Plot speed report."""
    for ax, vp, vr in zip(axs, vproc, vraw):
        ax.plot(tproc, vp, linewidth=1.0, label="Processed")
        ax.plot(traw, vr, linewidth=1.0, label="Raw")
    axs[-1].plot(
        tproc, np.linalg.norm(vproc, axis=0), linewidth=1.0, label="Processed")
    axs[-1].plot(
        traw, np.linalg.norm(vraw, axis=0), linewidth=1.0, label="Raw")
    axs[-1].axhline(dmax, linestyle='--', color='black')

    axs[0].set_ylabel("$v_x$ (m/s)")
    axs[1].set_ylabel("$v_y$ (m/s)")
    axs[2].set_ylabel("$v_z$ (m/s)")
    axs[3].set_ylabel("$||v||_2$ (m/s)")


def _main(args):

    if args.out is None:
        args.out = os.path.join(args.path, "_report", "speed.pdf")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    traj = rover.RawTrajectory.from_csv(
        os.path.join(args.path, "_slam", "trajectory.csv"))
    base_time = traj.t[0]
    duration = traj.t[-1] - base_time

    vraw = np.diff(traj.xyz, axis=1) / np.diff(traj.t)
    traw = (traj.t[1:] + traj.t[:-1]) / 2 - base_time

    radarpose = np.load(os.path.join(args.path, "_radar", "pose.npz"))
    vproc = radarpose["vel"].T
    tproc = radarpose["t"] - base_time

    with open(os.path.join(args.path, "radar", "radar.json")) as f:
        cfg = json.load(f)
        nd = cfg["shape"][0]
        dmax = cfg["doppler_resolution"] * (nd // 2)

    with PdfPages(args.out) as pdf:
        for i in tqdm(range(math.ceil(duration / args.width / 2))):
            fig, page = plt.subplots(8, 1, figsize=(8.5, 11))

            for j, axs in enumerate([page[:4], page[4:]]):
                _plot_speed(axs, tproc, traw, vproc, vraw, dmax)
                for ax in axs:
                    ax.set_xlim(
                        (i * 2 + j) * args.width, (i * 2 + j + 1) * args.width)
                    ax.grid()

            page[-1].set_xlabel("Time (s)")
            page[0].legend(
                loc='upper left', ncols=2, bbox_to_anchor=(-0.02, 1.4),
                frameon=False)
            fig.align_ylabels(page)
            fig.tight_layout(pad=5, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
