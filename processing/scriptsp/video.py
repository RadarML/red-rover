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

import imageio
import jax
import matplotlib as mpl
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


def _mask(
    img: Shaped[Array, "w h 3"], mask: Bool[Array, "w h"],
    color: tuple[int, int, int]
) -> Shaped[Array, "w h 3"]:
    return cast(Shaped[Array, "w h 3"], jnp.where(
        mask[:, :, None],
        jnp.array(color, dtype=np.uint8)[None, None, :], img))


def _transforms():
    """Initialize (and close on) transforms."""
    viridis = (
        jnp.array(mpl.colormaps['viridis'].colors) * 255   # type: ignore
    ).astype(jnp.uint8)
    greys = jnp.array([jnp.arange(255, dtype=np.uint8) for _ in range(3)]).T

    @jax.jit
    def rng_tf(x):
        x_norm = jnp.clip(x / 20000, 0, 1)
        img = graphics.lut(greys, x_norm)
        # No return
        img = _mask(img, x == 0, (199, 50, 40))
        # Infinite return
        img = _mask(img, x == 65536, (48, 199, 40))
        # >20m
        img = _mask(img, x > 20000, (40, 103, 199))
        return graphics.resize(img, 480, 1920)

    @jax.jit
    def lidar_tf(x):
        p1, p99 = jnp.percentile(x, jnp.array((1, 99)))
        x_norm = (x - p1) / p99
        return graphics.resize(graphics.lut(greys, x_norm), 480, 1920)

    @jax.jit
    def radar_tf(x):
        x = jnp.swapaxes(x, 0, 1)
        p1, p99 = jnp.percentile(x, jnp.array((1, 99.9)))
        x_norm = (jnp.clip(x, p1, p99) - p1) / p99
        return jax.vmap(
            partial(graphics.resize, width=480, height=720),
            in_axes=2, out_axes=2
        )(graphics.lut(viridis, x_norm))

    return {
        "radar": radar_tf, "camera": jnp.array,
        "rng": rng_tf, "rfl": lidar_tf, "nir": lidar_tf}


def _renderer(dataset_path, font_path):
    """Create (and close on) renderer."""
    font = graphics.JaxFont(font_path, size=80)

    @jax.jit
    def _render_frame(active, text):
        frame = (
            jnp.zeros((2160, 3840, 3), dtype=jnp.uint8)
            .at[360:360 + 1080, :1920].set(active["camera"])
            .at[:480, 1920:].set(active["rng"])
            .at[480:960, 1920:].set(active["rfl"])
            .at[960:1440, 1920:].set(active["nir"]))

        for i in range(8):
            frame = frame.at[1440:, i * 480:(i + 1) * 480].set(
                active["radar"][:, :, i])

        white = jnp.array([255, 255, 255], dtype=np.uint8)
        for k, v in text.items():
            frame = font.render(v, frame, white, x=k[0], y=k[1])

        return frame

    def render_frame(active, ii, dt):
        text =  {
            (40, 20): "red-rover",
            (40, 120): dataset_path[:18],
            (40, 220): "+{:02d}:{:05.2f}s".format(int(dt / 60), dt % 60),
            (1000, 20): "radar   {:06d}".format(ii["radar"]),
            (1000, 120): "lidar   {:06d}".format(ii["rng"]),
            (1000, 220): "camera  {:06d}".format(ii["camera"])
        }
        return np.array(_render_frame(
            active, {k: font.encode(v) for k, v in text.items()}))

    return render_frame


def _main(args):

    if args.out is None:
        args.out = os.path.join(args.path, "_report", "data.mp4")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    timestamps, streams = _load(args.path, args.radar)
    transforms = _transforms()
    render_func = _renderer(args.path, args.font)
    frame_time = 1 / args.fps * args.timescale

    ii = {k: 0 for k in timestamps}
    start_time = max(v[0] for v in timestamps.values()) // 1 + 1
    ts = start_time
    active = {k: next(v) for k, v in streams.items()}

    # Note: we don't use nvenc because the quality is terrible
    writer = imageio.get_writer(args.out, fps=args.fps, codec="h264")
    pbar = tqdm(total=sum(v.shape[0] for v in timestamps.values()))
    try:
        while True:
            # Advance index
            for k in timestamps:
                while timestamps[k][ii[k]] < ts:
                    ii[k] += 1
                    active[k] = transforms[k](next(streams[k]))
                    pbar.update(1)

            # Write frame
            frame = render_func(active, ii, ts - start_time)
            writer.append_data(frame)

            ts += frame_time
    except StopIteration:
        pass

    writer.close()
