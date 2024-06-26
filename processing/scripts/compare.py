"""Create simulation (novel view synthesis) comparison video.

Inputs: any set of radar-like data in `_radar`, or elsewhere so long as they
    match the format of `_radar/rda`.
Outputs: `_report/compare.mp4` unless overridden.
"""

import os
import imageio
from tqdm import tqdm
from functools import partial
import matplotlib as mpl

import jax
from jax import numpy as jnp
import numpy as np

from rover import graphics, Dataset, RawChannel


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-c", "--compare", nargs='+', default=['rda', 'sim_lidar'],
        help="Items to compare.")
    p.add_argument(
        "-o", "--out", default=None, help="Output path; defaults to "
        "`_report/compare.mp4` in the dataset folder.")
    p.add_argument(
        "--font", default=None, help="Use a specific `.ttf` font file.")


DEFAULT_NAMES = {
    "hybrid": "Measured / Hybrid Window",
    "hann": "Measured / Hanning Window",
    "raw": "Measured / Raw FFT",
    "sim_lidar": "Lidar Simulation",
    "sim_cfar": "CFAR Simulation",
    "sim_nearest": "Nearest Neighbor"
}


def _renderer(dataset_path, font_path, channel_names, n_frames):
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

    @jax.jit
    def _render_frame(frames, text):
        frame = jnp.zeros((480 * len(frames) + 96, 3840, 3), dtype=jnp.uint8)

        for i, img in enumerate(frames):
            img = radar_tf(img)
            for j in range(8):
                frame = frame.at[
                    i * 480 + 96:(i + 1) * 480 + 96, j * 480: (j + 1) * 480
                ].set(img[:, :, j])

        white = jnp.array([255, 255, 255], dtype=np.uint8)
        for k, v in text.items():
            frame = font.render(v, frame, white, x=k[0], y=k[1])

        return frame

    def render_frame(frames, i, t):
        text = {
            (20, 10): dataset_path[:18],
            (980, 10): "t+{:02d}:{:05.2f}s ({:02d}%)".format(
                int(t / 60), t % 60, int(100 * i / n_frames)),
            (1940, 10): "f+{:06d}/{:06d}".format(i, n_frames)
        }
        for i, cn in enumerate(channel_names):
            if cn in DEFAULT_NAMES:
                n = DEFAULT_NAMES[cn]
            else:
                n = cn.split('/')[-2]
            text[(20, 10 + 96 + 480 * i)] = n  # type: ignore
        return np.array(_render_frame(
            frames, {k: font.encode(v) for k, v in text.items()}))

    return render_frame


def _main(args):

    if args.out is None:
        args.out = os.path.join(args.path, "_report", "compare.mp4")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    radar = Dataset(args.path).get("_radar")

    ts = radar.timestamps()
    ts = ts - ts[0]
    fps = (ts.shape[0] - 1) / ts[-1]

    def _get_data(k):
        if k in radar.channels:
            return radar[k]
        else:
            return RawChannel(
                k, radar[args.compare[0]].type, radar[args.compare[0]].shape)

    data = [_get_data(k).stream_prefetch() for k in args.compare]
    render_func = _renderer(args.path, args.font, args.compare, ts.shape[0])

    writer = imageio.get_writer(args.out, fps=fps, codec="h264")            
    for i, (t, *frames) in enumerate(zip(tqdm(ts), *data)):
        writer.append_data(render_func(frames, i, t))

    writer.close()
