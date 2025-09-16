"""Video anonymization."""

import os
from typing import cast

import cv2
from roverd import channels, sensors
from roverd.channels.utils import Prefetch
from tqdm import tqdm


def cli_anonymize(
    path: str, /, out: str | None = None
) -> None:
    """Anonymize video by blurring faces.

    !!! warning

        Requires the `anonymize` extra (`retina-face`, `tf-keras`).

    !!! io "Expected Inputs and Outputs"

        **Inputs**: `camera/video.avi`

        **Outputs**: `_camera/video.avi`"

    Args:
        path: path to the dataset.
        out: output path; defaults to the same as `path`.
    """
    from retinaface import RetinaFace

    camera = sensors.Camera(os.path.join(path, "camera"))

    if out is None:
        out = path

    _camera = sensors.DynamicSensor(
        os.path.join(out, "_camera"), create=True, exist_ok=True)
    output = _camera.create("video.avi", camera.config["video.avi"])

    def _apply_image(image):
        """Apply face blurring to an image."""
        faces = RetinaFace.detect_faces(image)

        # If no faces detected, return original image
        if not isinstance(faces, dict) or len(faces) == 0:
            return image

        # Apply blur to each detected face
        for face_key, face_data in faces.items():
            if 'facial_area' in face_data:
                x1, y1, x2, y2 = face_data['facial_area']

                # Expand by 20%
                width = x2 - x1
                height = y2 - y1
                x1 -= int(0.2 * width)
                y1 -= int(0.2 * height)
                x2 += int(0.2 * width)
                y2 += int(0.2 * height)

                # Ensure coordinates are within image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image.shape[1], int(x2))
                y2 = min(image.shape[0], int(y2))

                # Extract face region
                face_region = image[y1:y2, x1:x2]

                # Apply Gaussian blur to the face region
                if face_region.size > 0:
                    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 15)
                    image[y1:y2, x1:x2] = blurred_face

        return image

    stream = camera["video.avi"].stream_prefetch()
    frame_stream = Prefetch(
        _apply_image(frame) for frame in tqdm(stream, total=len(camera))
    ).queue
    cast(channels.VideoChannel, output).consume(frame_stream)
