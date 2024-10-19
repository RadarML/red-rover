"""Run semantic segmentation on collected video data.

NOTE: requires `transformers` to be installed, along with dependencies::

    pip install torch torchvision transformers

These dependencies are not installed by `make env` by default, since they are
rather heavy and used only for this script.

By default, we use `segformer-b5-finetuned-ade-640-640`, which can be found
`here <https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640>`__;
the `pytorch_model.bin` weights should be downloaded and placed in
`processing/models/segformer-b5-finetuned-ade-640-640`. The original `class
definitions <https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv>`__
have been reduced into 8 classes (indexed in alphabetical order), as specified in
`models/segformer-b5-finetuned-ade-640-640/classes.yaml`:

- `0=ceiling`: ceiling, roofs, etc viewed from underneath.
- `1=flat`: flat ground such as roads, paths, floors, etc.
- `2=nature`: plants of all kinds such as trees, grass, and bushes.
- `3=object`: any free-standing objects other than plants which are not
  building-scale such as furniture, signs, and poles.
- `4=person`: any person who is not inside a vehicle.
- `5=sky`: the sky or other background.
- `6=structure`: buildings, fences, and other structures.
- `7=vehicle`: cars, busses, vans, trucks, etc.

Inputs:
    - `camera/*`

Outputs:
    - `_camera/segment`
"""

import os

import numpy as np
import yaml
from beartype.typing import cast
from jaxtyping import UInt8
from roverd import Dataset, channels
from tqdm import tqdm


def _get_classmap(path: str) -> UInt8[np.ndarray, "8"]:
    with open(os.path.join(path, "classes.yaml")) as f:
        meta = yaml.load(f, Loader=yaml.SafeLoader)

    orig = {k: i for i, k in enumerate(meta["original"])}
    classmap = np.zeros(len(orig), dtype=np.uint8)
    for i, classname in enumerate(sorted(meta["reduced"].keys())):
        for oldclass in meta["reduced"][classname]:
            classmap[orig[oldclass]] = i
    return classmap


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument("-b", "--batch", type=int, default=16, help="Batch size.")
    p.add_argument(
        "--model", default="segformer-b5-finetuned-ade-640-640",
        help="Model to use.")


def _main(args):

    # Ignore for type checking since torch & transformers aren't shipped by
    # default.
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
    except ImportError:
        raise Exception(
            "Must have `torch` and `transformers` installed (not included in "
            "default installation).")

    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", args.model)
    feature_extractor = (
        transformers.SegformerImageProcessor.from_pretrained(path))
    model = transformers.SegformerForSemanticSegmentation.from_pretrained(
        path).to('cuda')  # type: ignore
    classmap = torch.from_numpy(_get_classmap(path)).to('cuda')

    def _apply_image(imgs):
        with torch.no_grad(), torch.amp.autocast('cuda'):  # type: ignore
            inputs = feature_extractor(  # type: ignore
                images=imgs, return_tensors='pt').to('cuda')
            outputs = model(**inputs)
            classes_raw = torch.argmax(outputs.logits, dim=1)
        classes_reduced = classmap[classes_raw]
        return classes_reduced.cpu().numpy()

    ds = Dataset(args.path)
    camera = ds["camera"]
    _camera = ds.virtual_copy("camera", exist_ok=True)

    output = _camera.create("segment", {
        "format": "lzmaf", "type": "u1", "shape": [160, 160],
        "desc": "Image segmentation with 640x640 resize and 4x downsample."})
    stream = camera["video.avi"].stream_prefetch(batch=args.batch)

    frame_stream = (
        _apply_image(frame) for frame in
        tqdm(stream, total=int(np.ceil(len(camera) / args.batch))))

    # frame_stream is already batched
    cast(channels.LzmaFrameChannel, output).consume(frame_stream, batch=0)
