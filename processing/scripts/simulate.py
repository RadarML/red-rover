"""Simulate radar range-doppler data.

Inputs: `_radar/pose.npz`, `_slam/map.npz`
Outputs: `_radar/sim_lidar` or `_radar/sim_cfar`
"""

import os
from tqdm import tqdm
from functools import partial

import jax
from jax import numpy as jnp

from rover import Dataset


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument("-k", "--key", type=int, default=42, help="Random seed.")
    p.add_argument(
        "-m", "--mode", default="lidar", help="Simulation mode (lidar/cfar).")


def _main(args):

    try:
        from arrow import ReflectanceGrid, Arrow, rover_intrinsics, types
    except ImportError:
        print("Must have arrow installed to run simulation.")
        exit(1)

    intrinsics = rover_intrinsics(path=args.path, backend=jnp)
    num_doppler = intrinsics.doppler.shape[0]
    grid, params = ReflectanceGrid.from_rover(
        path=args.path, backend=jnp, mode=args.mode)
    arrow = Arrow(backend=jnp, k=128, eps=1e-5)

    @jax.jit
    def _render(key, params, pose):
        field = partial(grid, grid=params)
        return arrow.render(
            offsets=jax.random.uniform(key, shape=(num_doppler,)),
            doppler=jnp.arange(num_doppler, dtype=jnp.uint8),
            pose=pose, intrinsics=intrinsics, field=field)

    pose = arrow.make_pose(types.RawPose.from_npz(
        os.path.join(args.path, "_radar", "pose.npz")))
    seeds = jax.random.split(jax.random.PRNGKey(args.key), pose.x.shape[0])

    def render(i):
        return _render(seeds[i], params, pose[i:i+1])

    out = Dataset(args.path).get("_radar").create(f"sim_{args.mode}", {
        "format": "raw", "type": "f4",
        "shape": [intrinsics.doppler.shape[0], intrinsics.range.shape[0], 8],
        "desc": "Simulated doppler-range-azimuth image using lidar data."})
    out.consume((render(i) for i in tqdm(range(pose.x.shape[0]))))
