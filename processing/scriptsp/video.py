"""Render sensor data video.

Inputs:
    - `camera/*`
    - `lidar/*`
    - `_radar/{radar}` for the specified `--radar`.

Outputs:
    - `_report/data.mp4`
"""

import os
from functools import partial

import jax
import numpy as np
from beartype.typing import cast
from jax import numpy as jnp
from jaxtyping import Array, Bool, Shaped
from roverd import Dataset, sensors
from tqdm import tqdm

from roverp import graphics


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out", default=None, help="Output path; defaults to "
        "`_report/data.mp4` in the dataset folder.")
    p.add_argument(
        "--font", default=None, help="Use a specific `.ttf` font file.")
    p.add_argument(
        "-f", "--fps", type=float, default=30.0,
        help="Output video framerate.")
    p.add_argument(
        "-s", "--timescale", type=float, default=1.0,
        help="Real time to video time scale factor (larger = faster)")
    p.add_argument(
        "-r", "--radar", default="hybrid", help="Radar stream to use.")


def _load(path: str, radar: str):
    """Load timestamps and streams."""
    ds = Dataset(path)

    radar_meta = 'radar' if radar == 'raw' else '_radar'

    _ts = {k: ds[k].timestamps() for k in [radar_meta, "lidar", "camera"]}
    timestamps = {
        "radar": _ts[radar_meta], "camera": _ts["camera"],
        "rfl": _ts["lidar"], "nir": _ts["lidar"], "rng": _ts["lidar"]}

    streams = {
        "radar": ds["_radar"][radar].stream_prefetch(),
        "rng": cast(sensors.LidarData, ds["lidar"]).destaggered_stream("rng"),
        "rfl": cast(sensors.LidarData, ds["lidar"]).destaggered_stream("rfl"),
        "nir": cast(sensors.LidarData, ds["lidar"]).destaggered_stream("nir"),
        "camera": ds["camera"]["video.avi"].stream_prefetch()
    }

    return timestamps, streams


def _renderer(dataset_path, font_path):
    """Create (and close on) renderer."""
    viridis = graphics.mpl_colormap('viridis')
    greys = jnp.array([jnp.arange(255, dtype=np.uint8) for _ in range(3)]).T

    def _mask(
        img: Shaped[Array, "w h 3"], mask: Bool[Array, "w h"],
        color: tuple[int, int, int]
    ) -> Shaped[Array, "w h 3"]:
        return cast(Shaped[Array, "w h 3"], jnp.where(
            mask[:, :, None],
            jnp.array(color, dtype=np.uint8)[None, None, :], img))

    def rng_tf(x):
        img = graphics.render_image(x, greys, scale=20000)
        # No return
        img = _mask(img, x == 0, (199, 50, 40))
        # Infinite return
        img = _mask(img, x == 65536, (48, 199, 40))
        # >20m
        img = _mask(img, x > 20000, (40, 103, 199))
        return graphics.resize(img, 480, 1920)

    def lidar_tf(x):
        return graphics.render_image(
            x, greys, pmin=1.0, pmax=99.0, resize=(480, 1920))

    def radar_tf(x):
        # doppler-range-azimuth -> azimuth-range-doppler
        x = jnp.moveaxis(x, (0, 1, 2), (2, 1, 0))

        stack = jax.vmap(partial(
            graphics.render_image, colors=viridis,
            resize=(720, 480), pmin=1.0, pmax=99.9
        ))(x)
        return jnp.concatenate(list(stack), axis=1)

    return graphics.Render(
        size=(2160, 3840),
        channels={
            (360, 360 + 1080, 0, 1920): "camera",
            (0, 480, 1920, 3840): "rng",
            (480, 960, 1920, 3840): "rfl",
            (960, 1440, 1920, 3840): "nir",
            (1440, 2160, 0, 3840): "radar"
        },
        transforms={
            "radar": radar_tf, "rng": rng_tf, "rfl": lidar_tf, "nir": lidar_tf
        },
        text={
            (20, 40): "red-rover",
            (120, 40): dataset_path[:18],
            (220, 40): "+{mm:02d}:{ss:05.2f}s",
            (20, 1000): "radar   {radar:06d}",
            (120, 1000): "lidar   {rng:06d}",
            (220, 1000): "camera  {camera:06d}"
        },
        font=graphics.JaxFont(font_path, size=80),
        textcolor=(255, 255, 255)
    ).render


def _main(args):

    if args.out is None:
        args.out = os.path.join(args.path, "_report", "data.mp4")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    timestamps, streams = _load(args.path, args.radar)
    render_func = _renderer(args.path, args.font)
    frame_time = 1 / args.fps * args.timescale

    est_n_frames = int((
        min(v[-1] for v in timestamps.values())
        - max(v[0] for v in timestamps.values())
    ) / frame_time)

    synced = graphics.synchronize(
        streams, timestamps, period=frame_time, round=1.0)
    render_iter = (
        render_func(frameset, {"mm": int(t / 60), "ss": (t % 60), **ii})
        for t, ii, frameset in tqdm(synced, total=est_n_frames))

    graphics.write_consume(
        render_iter, out=args.out, fps=args.fps, codec="h264")
