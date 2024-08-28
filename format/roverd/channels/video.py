"""Video (nominally mjpeg) channel."""

import cv2
import numpy as np
from jaxtyping import Shaped
from beartype.typing import Iterator, Callable, Optional, Any

from .base import Channel


class VideoChannel(Channel):
    """Video data.

    NOTE: while opencv is a heavy dependency, it seems to have very efficient
    seeking for mjpeg compared to `imageio`, the other library that I tested.
    Using `opencv-python-headless` instead of the default opencv should
    alleviate some of these issues.
    """

    def read(
        self, start: int = 0, samples: int = -1
    ) -> Shaped[np.ndarray, "samples ..."]:
        """Read data.

        Args:
            start: start index to read.
            samples: number of samples/frames to read. If `-1`, read all data.

        Returns:
            Read frames as an array, with a leading axis corresponding to
            the number of `samples`. If only a subset of frames are readable
            (e.g. due to reaching the end of the video), the result is
            truncated.

        Raises:
            ValueError: None of the frames could be read, possibly due to
                an invalid video, or invalid start index.
        """
        cap = cv2.VideoCapture(self.path)

        if start != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)

        frames: list[np.ndarray] = []
        while cap.isOpened():
            ret, frame = cap.read()
            # if `samples == -1`, this is never satisfied.
            if ret and len(frames) != samples:
                frames.append(frame[..., ::-1])
            else:
                break

        cap.release()
        if len(frames) == 0:
            raise ValueError("Could not read any frames.")
        return np.stack(frames)

    def stream(
        self, transform: Optional[
            Callable[[Shaped[np.ndarray, "..."]], Any]
        ] = None, batch: int = 0
    ) -> Iterator[np.ndarray]:
        """Get iterable data stream.

        Args:
            transform: callable to apply to the read data. Should take a single
                sample or batch of samples, and can return an arbitrary type.
            batch: batch size to read. If 0, load only a single sample and do
                not append an empty axis.

        Returns:
            Iterator which yields successive frames.
        """
        if transform is None:
            transform = lambda x: x

        cap = cv2.VideoCapture(self.path)
        frames: list[np.ndarray] = []
        while cap.isOpened():
            if len(frames) == batch:
                yield transform(np.stack(frames))

            ret, frame = cap.read()
            if ret:
                if batch == 0:
                    yield transform(frame[..., ::-1])
                else:
                    frames.append(frame[..., ::-1])
            else:
                break

        if len(frames) > 0:
            yield transform(np.stack(frames))

        cap.release()
        return
