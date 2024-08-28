"""Create clip embeddings for each frame in the video.

NOTE: requires `open_clip` to to be installed, along with dependencies::

    pip install torch torchvision open_clip_torch==2.26.1

These dependencies are not installed by `make env` by default, since they are
rather heavy and used only for this script.

Models and checkpoints for `open_clip` can be found
`here <https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv>`_.
The model selected will be downloaded the first time it is used.

Inputs:
    - `camera/*`

Outputs:
    - `_camera/clip`
"""

import cv2
import numpy as np
from einops import rearrange
from tqdm import tqdm

from rover import Dataset


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument("--model", default="ViT-L-14", help="Model to use.")
    p.add_argument(
        "--checkpoint", default="datacomp_xl_s13b_b90k",
        help="Checkpoint to use.")

def _main(args):

    # Ignore errors here since torch & open_clip aren't shipped by default.
    try:
        import open_clip  # type: ignore
        import torch      # type: ignore

        mean = np.array(open_clip.constants.OPENAI_DATASET_MEAN)
        std = np.array(open_clip.constants.OPENAI_DATASET_STD)

    except ImportError:
        raise Exception(
            "Must have `torch` and `open_clip_torch` installed (not included"
            "in default installation).")

    # We do our own preprocessing since open_clip's preprocessing is a
    # complete mess that doesn't allow batching and forces going through PIL
    model = open_clip.create_model(
        args.model, pretrained=args.checkpoint, device='cuda')
    model.eval()

    def _preprocess(img):
        resized = cv2.resize(
            img, (224 * 4, 224 * 2), interpolation=cv2.INTER_CUBIC)
        chunks = rearrange(
            resized, "(h blkh) (w blkw) c -> (h w) c blkh blkw",
            h=2, w=4, blkw=224, blkh=224)
        return ((
            chunks.astype(np.float32) - mean[None, :, None, None]
        ) / std[None, :, None, None]).astype(np.float16)

    def _apply_image(img):
        with torch.no_grad(), torch.cuda.amp.autocast():
            features = model.encode_image(torch.Tensor(img).to('cuda'))
        return rearrange(features.cpu().numpy(), "(h w) c -> h w c", h=2, w=4)

    ds = Dataset(args.path)
    camera = ds["camera"]
    _camera = ds.create("_camera", exist_ok=True)

    ts = _camera.create("ts", camera.config["ts"])
    ts.write(camera["ts"].read())

    embeddings = _camera.create("clip", {
        "format": "raw", "type": "f2", "shape": [2, 4, 768],
        "desc": "Clip embeddings, applied to 8 patches (4 wide by 2 high)."})
    stream = camera["video.avi"].stream_prefetch(transform=_preprocess)
    embeddings.consume(
        _apply_image(frame) for frame in tqdm(stream, total=len(camera)))
