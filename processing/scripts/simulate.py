"""Simulate radar range-doppler data."""

import os
from tqdm import tqdm

import numpy as np
import jax
from jax import numpy as jnp

from arrow import ReflectanceGrid, Arrow, rover_intrinsics
from rover import Dataset


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument("-k", "--key", type=int, default=42, help="Random seed.")


def _main(args):

    intrinsics = rover_intrinsics(path=args.path, backend=jnp)
    num_doppler = intrinsics.doppler.shape[0]
    grid, params = ReflectanceGrid.from_rover(path=args.path, backend=jnp)
    arrow = Arrow(backend=jnp, k=128, eps=1e-5)

    @jax.jit
    def _render(key, params, pos, vel, rot):
        pose = arrow.make_pose(pos, vel, rot)
        return arrow.render(
            offsets=jax.random.uniform(key, shape=(num_doppler,)),
            doppler=jnp.arange(num_doppler, dtype=jnp.uint8),
            pose=pose, intrinsics=intrinsics, params=params, field=grid)

    npz = np.load(os.path.join(args.path, "_radar", "pose.npz"))
    pos = npz["pos"]
    vel = npz["vel"]
    rot = npz["rot"]

    seeds = jax.random.split(jax.random.PRNGKey(args.key), pos.shape[0])

    def render(i):
        return _render(seeds[i], params, pos[i:i+1], vel[i:i+1], rot[i:i+1])

    out = Dataset(args.path).get("_radar").create("sim_lidar", {
        "format": "raw", "type": "f32",
        "shape": [intrinsics.doppler.shape[0], intrinsics.range.shape[0], 8],
        "desc": "Simulated doppler-range-azimuth image using lidar data."})
    out.consume((render(i) for i in tqdm(range(pos.shape[0]))))
