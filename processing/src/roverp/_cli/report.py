"""Get speed report."""

import json
import math
import os
from typing import cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def cli_report(
    path: str, /, out: str | None = None, width: float = 30.0
) -> None:
    """Generate speed report.

    !!! warning

        The cartographer slam pipeline (`make trajectory` or `make lidar`) must
        be run beforehand.

    !!! io "Expected Inputs and Outputs"

        **Inputs**: `_radar/pose.npz`, `_slam/trajectory.csv`

        **Outputs**: `_report/speed.pdf`

    Args:
        path: path to the dataset.
        out: output path; defaults to `_report/speed.pdf` in the dataset.
        width: time series plot row width, in seconds.
    """
    from roverp.readers import RawTrajectory

    def _plot_speed(axs, tproc, traw, vproc, vraw, dmax):
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

    if out is None:
        out = os.path.join(path, "_report", "speed.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    trajfile = os.path.join(path, "_slam", "trajectory.csv")
    if not os.path.exists(trajfile):
        raise FileNotFoundError(
            f"Trajectory file not found: {trajfile}. "
            f"Run `make trajectory` or `make lidar` first.")

    traj = RawTrajectory.from_csv(trajfile)
    base_time = traj.t[0]
    duration = traj.t[-1] - base_time

    vraw = np.diff(traj.xyz, axis=1) / np.diff(traj.t)
    traw = (traj.t[1:] + traj.t[:-1]) / 2 - base_time  # type: ignore

    radarpose = np.load(os.path.join(path, "_radar", "pose.npz"))
    vproc = radarpose["vel"].T
    tproc = radarpose["t"] - base_time

    with open(os.path.join(path, "radar", "radar.json")) as f:
        cfg = json.load(f)
        nd = cfg["shape"][0]
        dmax = cfg["doppler_resolution"] * (nd // 2)

    with PdfPages(out) as pdf:
        for i in tqdm(range(math.ceil(duration / width / 2))):
            fig, page = plt.subplots(8, 1, figsize=(8.5, 11))
            page = cast(np.ndarray, page)

            for j, axs in enumerate([page[:4], page[4:]]):
                _plot_speed(axs, tproc, traw, vproc, vraw, dmax)
                for ax in axs:
                    ax.set_xlim(
                        (i * 2 + j) * width, (i * 2 + j + 1) * width)
                    ax.grid()

            page[-1].set_xlabel("Time (s)")
            page[0].legend(
                loc='upper left', ncols=2, bbox_to_anchor=(-0.02, 1.4),
                frameon=False)
            fig.align_ylabels(page)
            fig.tight_layout(pad=5, h_pad=0.2, w_pad=0.2)
            pdf.savefig(fig)
