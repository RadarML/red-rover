"""Run semantic segmentation on collected video data."""

import os
from typing import cast

import numpy as np
import yaml
from jaxtyping import UInt8
from roverd import channels, sensors
from tqdm import tqdm


def cli_segment(
    path: str, /, batch: int = 16,
    model: str = "./models/segformer-b5-finetuned-ade-640-640",
) -> None:
    """Run semantic segmentation on collected video data.

    !!! warning

        Requires the `semseg` extra (`torch`, `transformers`):
        ```sh
        pip install ./processing[semseg]
        # equivalent to
        pip install torch torchvision transformers
        ```

    By default, we use `segformer-b5-finetuned-ade-640-640`, which can be found
    [here](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640);
    the `pytorch_model.bin` weights should be downloaded and placed in
    `processing/models/segformer-b5-finetuned-ade-640-640`.

    ??? info "Class Definitions"

        The original [class definitions](
        https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv)
        have been reduced into 8 classes (indexed in alphabetical order), as
        specified in `models/segformer-b5-finetuned-ade-640-640/classes.yaml`:

        - `0=ceiling`: ceiling, roofs, etc viewed from underneath.
        - `1=flat`: flat ground such as roads, paths, floors, etc.
        - `2=nature`: plants of all kinds such as trees, grass, and bushes.
        - `3=object`: any free-standing objects other than plants which are not
            building-scale such as furniture, signs, and poles.
        - `4=person`: any person who is not inside a vehicle.
        - `5=sky`: the sky or other background.
        - `6=structure`: buildings, fences, and other structures.
        - `7=vehicle`: cars, busses, vans, trucks, etc.

    !!! io "Expected Inputs and Outputs"

        **Inputs**: `camera/video.avi`

        **Outputs**: `_camera/segment`

    Args:
        path: path to the recording.
        batch: batch size for processing; may need tuning, depending on your
            exact GPU.
        model: path to model weights; if running in the `red-rover/processing`
            directory, this should be
            `./models/segformer-b5-finetuned-ade-640-640`.
    """
    # Ignore for type checking since torch & transformers aren't shipped by
    # default.
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
    except ImportError:
        raise Exception(
            "Must have `torch` and `transformers` installed (not included in "
            "default installation).")

    def _get_classmap(path: str) -> UInt8[np.ndarray, "8"]:
        with open(os.path.join(path, "classes.yaml")) as f:
            meta = yaml.load(f, Loader=yaml.SafeLoader)

        orig = {k: i for i, k in enumerate(meta["original"])}
        classmap = np.zeros(len(orig), dtype=np.uint8)
        for i, classname in enumerate(sorted(meta["reduced"].keys())):
            for oldclass in meta["reduced"][classname]:
                classmap[orig[oldclass]] = i
        return classmap

    feature_extractor = (
        transformers.SegformerImageProcessor.from_pretrained(model))
    torch_model = transformers.SegformerForSemanticSegmentation.from_pretrained(
        model).to('cuda')  # type: ignore
    classmap = torch.from_numpy(_get_classmap(model)).to('cuda')

    def _apply_image(imgs):
        with torch.no_grad(), torch.amp.autocast('cuda'):  # type: ignore
            inputs = feature_extractor(  # type: ignore
                images=imgs, return_tensors='pt').to('cuda')
            outputs = torch_model(**inputs)
            classes_raw = torch.argmax(outputs.logits, dim=1)
        classes_reduced = classmap[classes_raw]
        return classes_reduced.cpu().numpy()

    camera = sensors.DynamicSensor(os.path.join(path, "camera"))

    _camera = sensors.DynamicSensor(
        os.path.join(path, "_camera"), create=True, exist_ok=True)
    output = _camera.create("segment", {
        "format": "lzmaf", "type": "u1", "shape": [160, 160],
        "desc": "Image segmentation with 640x640 resize and 4x downsample."})
    stream = camera["video.avi"].stream_prefetch(batch=batch)

    frame_stream = (
        _apply_image(frame) for frame in
        tqdm(stream, total=int(np.ceil(len(camera) / batch))))

    # frame_stream is already batched
    cast(channels.LzmaFrameChannel, output).consume(frame_stream, batch=0)
