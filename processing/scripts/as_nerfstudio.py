"""Convert poses and video into Nerfstudio format.

See the `Nerfstudio data conventions:
<https://docs.nerf.studio/quickstart/data_conventions.html>`_.

Inputs:
    - `calibration/camera_ns.json`
    - `camera/video.avi`
    - `_camera/pose.npz`

Outputs:
    - `_nerfstudio`
"""

import os, json
import numpy as np
import cv2
from tqdm import tqdm

from rover import Dataset


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-s", "--stride", type=int, default=30,
        help="Use every `--stride` frames.")
    p.add_argument(
        "-i", "--intrinsics", default="calibration/camera_ns.json",
        help="Camera intrinsics in nerfstudio format.")
    return p


def _main(args):
    cam = Dataset(args.path)["camera"]
    video = cam["video.avi"]

    with open(args.intrinsics) as f:
        intrinsics = json.load(f)

    out_path = os.path.join(args.path, "_nerfstudio")
    os.makedirs(os.path.join(out_path, "images"), exist_ok=True)

    poses = np.load(os.path.join(args.path, "_camera", "pose.npz"))

    # Indices
    keep_frames = np.arange(0, poses['t'].shape[0], args.stride)
    rot = poses['rot'][keep_frames]
    pos = poses['pos'][keep_frames]

    raw_indices = np.arange(len(cam))[poses['mask']][keep_frames]

    # Coordinate transform
    c2w = np.concatenate([
        # Rotation, position
        np.stack([-rot[:, :, 1], rot[:, :, 2], -rot[:, :, 0], pos], axis=-1),
        # [0 0 0 1]
        np.broadcast_to(
            np.array([0, 0, 0, 1])[None, None, :], (rot.shape[0], 1, 4))
    ], axis=1)

    intrinsics["frames"] = []
    for i, pose in zip(tqdm(raw_indices), c2w):
        frame_path = os.path.join("images", "frame_{}.jpg".format(i))
        cv2.imwrite(os.path.join(out_path, frame_path), video.index(i))
        intrinsics["frames"].append({
            "file_path": frame_path, "transform_matrix": pose.tolist()})

    with open(os.path.join(out_path, "transforms.json"), 'w') as f:
        json.dump(intrinsics, f)
