"""Render comparison video with camera reference.

Inputs:
    - any set of radar-like data in `_radar`, or elsewhere so long as they
      match the format of `_radar/rda`.

Outputs:
    - `_report/compare.mp4` unless overridden.
"""

import os
from functools import partial

import jax
import matplotlib as mpl
import numpy as np
from jax import numpy as jnp
from roverd import Dataset
from tqdm import tqdm

from roverp import graphics


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument("-c", "--compare", help="Path to simulated data.")
    p.add_argument(
        "-o", "--out", default=None, help="Output path; defaults to "
        "`_report/compare.mp4` in the dataset folder.")
    p.add_argument(
        "--font", default=None, help="Use a specific `.ttf` font file.")


def _renderer(dataset_path, font_path, n_frames):
    """Create (and close on) renderer."""
    font = graphics.JaxFont(font_path, size=60)
    viridis = (
        jnp.array(mpl.colormaps['viridis'].colors) * 255   # type: ignore
    ).astype(jnp.uint8)

    def radar_tf(x):
        x = jnp.swapaxes(x, 0, 1)
        p1, p99 = jnp.percentile(x, jnp.array((1, 99.9)))
        x_norm = (jnp.clip(x, p1, p99) - p1) / p99
        return jax.vmap(
            partial(graphics.resize, width=480, height=480),
            in_axes=2, out_axes=2
        )(graphics.lut(viridis, x_norm))

    def camera_tf(x):
        return graphics.resize(x[156:-156], width=3000, height=1200)

    @jax.jit
    def _render_frame(frames, text):
        frame = jnp.zeros(
            (2160, 3840, 3), dtype=jnp.uint8
        ).at[0:1200, 840:3840].set(camera_tf(frames["camera"]))

        for j in range(8):
            frame = frame.at[1200:1680, 480 * j: 480 * (j + 1)].set(
                radar_tf(frames["gt"][:, :, j])
            ).at[1680:2160, 480 * j: 480 * (j + 1)].set(
                radar_tf(frames["pred"][:, :, j]))

        white = jnp.array([255, 255, 255], dtype=np.uint8)
        for k, v in text.items():
            frame = font.render(v, frame, white, x=k[0], y=k[1])

        return frame

    def render_frame(frames, ii, t):
        text = {
            (40, 40): dataset_path[:18],
            (40, 140): "val (20%)",
            (40, 240): "t+{:02d}:{:05.2f}s".format(int(t / 60), t % 60),
            (40, 340): "f+{:04d}/{:04d}".format(ii["gt"], n_frames),
            (40, 1200 + 10): "Ground Truth",
            (40, 1200 + 480 + 10): "DART",
        }
        for j in range(8):
            text[(160 + 480 * j, 2060)] = f"TX{j // 4 + 1}/RX{j % 4 + 1}"

        return np.array(_render_frame(
            frames, {k: font.encode(v) for k, v in text.items()}))

    return render_frame


def _main(args):

    if args.out is None:
        args.out = os.path.join(args.path, "_report", "compare.mp4")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ds = Dataset(args.path)

    # Crop to validation split
    ts = ds["_radar"].timestamps()
    ts = ts[-int(0.2 * ts.shape[0]):]
    est_n_frames = int((ts[-1] - ts[0]) * 30)

    timestamps = {
        "gt": ts, "pred": ts, "camera": ds["camera"].timestamps()}
    streams = {
        "gt": ds["_radar"]["hybrid"].stream_prefetch(),
        "pred": ds["_radar"]["hybrid"].open_like(
            args.compare).stream_prefetch(),
        "camera": ds["camera"]["video.avi"].stream_prefetch()}

    render_func = _renderer(args.path, args.font, ts.shape[0])

    synced = graphics.synchronize(
        streams, timestamps, period=1 / 30, round=1.0)
    render_iter = (
        render_func(frameset, ii, t)
        for t, ii, frameset in tqdm(synced, total=est_n_frames))

    graphics.write_consume(render_iter, out=args.out, fps=30, codec="h264")
