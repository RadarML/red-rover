"""Render map slices.

Inputs:
    - any map-like data, e.g. `_slam/map.npz`.

Outputs:
    - `_report/slices.mp4` unless overridden.
"""

import os

import imageio
import jax
import matplotlib as mpl
import numpy as np
from jax import numpy as jnp
from tqdm import tqdm

from roverp import graphics


def _parse(p):
    p.add_argument("-p", "--path", help="Path to dataset or map.")
    p.add_argument(
        "-o", "--out", default=None, help="Output path; defaults to "
        "`_report/slices.mp4` in the dataset folder.")
    p.add_argument(
        "-f", "--fps", type=float, default=25.0,
        help="Output video framerate (i.e. layers/sec).")
    p.add_argument(
        "--font", default=None, help="Use a specific `.ttf` font file.")
    p.add_argument("--cmap", default="viridis", help="Colormap to use.")
    p.add_argument(
        "--mode", default="full",
        help="Rendering mode (only plots sigma & alpha when `mode == full`)")


def normalize(arr, left, right):
    """Apply frame normalization / auto-gain."""
    left, right = jnp.percentile(arr, jnp.array([left, right]))
    return (jnp.clip(arr, left, right) - left) / (right - left)


def _main(args):

    if args.out is None:
        args.out = os.path.join(args.path, "_report", "slices.mp4")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if os.path.isdir(args.path):
        args.path = os.path.join(args.path, "_slam", "map.npz")

    font = graphics.JaxFont(args.font, size=40)
    cmap = (
        jnp.array(mpl.colormaps[args.cmap].colors) * 255   # type: ignore
    ).astype(jnp.uint8)

    npz = np.load(args.path)
    if "grid" in npz and "shape" in npz:
        _size = np.prod(npz["shape"])
        data = jnp.rollaxis(
            np.unpackbits(npz["grid"])[:_size].reshape(npz["shape"]), 2)
        nz = data.shape[0]

        def _render_data(x):
            return jnp.take(jnp.stack([cmap[0], cmap[-1]]), x, axis=0)
    elif "grid" in npz:
        data = jnp.rollaxis(normalize(npz["grid"], 1, 99.5), 2)
        nz = data.shape[0]

        def _render_data(x):
            return graphics.lut(cmap, x)
    else:
        sigma = jnp.rollaxis(normalize(npz["sigma"], 1, 99.5), 2)
        alpha = jnp.rollaxis(normalize(-jnp.array(npz["alpha"]), 5, 99.5), 2)

        if args.mode == 'full':
            data = zip(sigma, alpha)
            nz = sigma.shape[0]

            def _render_data(x):
                sigma, alpha = x
                return jnp.concatenate(
                    [graphics.lut(cmap, sigma), graphics.lut(cmap, alpha)],
                    axis=1)
        else:
            data = sigma
            nz = sigma.shape[0]

            def _render_data(x):
                return graphics.lut(cmap, x)

    zz = np.linspace(npz["lower"][2], npz["upper"][2], nz)

    @jax.jit
    def _render_frame(data, text):
        frame = _render_data(data)
        white = jnp.array([255, 255, 255], dtype=np.uint8)
        for k, v in text.items():
            frame = font.render(v, frame, white, x=k[0], y=k[1])
        return frame

    writer = imageio.get_writer(args.out, fps=args.fps, codec="h264")
    for i, (z, x) in enumerate(zip(tqdm(zz), data)):
        text = {(40, 20): font.encode("{}/{}: {:+.2f}m".format(i + 1, nz, z))}
        writer.append_data(np.array(_render_frame(x, text)))
    writer.close()
