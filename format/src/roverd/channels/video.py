"""Video (nominally mjpeg) channel."""

from functools import cached_property
from queue import Queue
from threading import Thread

import numpy as np
from beartype.typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    cast,
)
from jaxtyping import Shaped

from .base import Buffer, Channel, Data, Streamable


class VideoChannel(Channel):
    """Video data.

    NOTE: while opencv is a heavy dependency, it seems to have very efficient
    seeking for mjpeg compared to `imageio`, the other library that I tested.
    Using `opencv-python-headless` instead of the default opencv should
    alleviate some of these issues.
    """

    @cached_property
    def _cv2_module(self):
        try:
            import cv2
            return cv2
        except ImportError:
            raise ImportError(
                "Could not import cv2. `opencv-python` or "
                "`opencv-python-headless` must be installed in order to use "
                "video encoding or decoding.")

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
        cap = self._cv2_module.VideoCapture(self.path)

        if start != 0:
            cap.set(self._cv2_module.CAP_PROP_POS_FRAMES, start - 1)

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

        cap = self._cv2_module.VideoCapture(self.path)
        frames: list[np.ndarray] = []
        while cap.isOpened():
            if batch != 0 and len(frames) == batch:
                yield transform(np.stack(frames))
                frames = []

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

    def consume(
        self, stream: Streamable[Data | Sequence[Data]],
        thread: bool = False, fps: float = 10.0
    ) -> None:
        """Consume iterable or queue and write to file.

        - If `Iterable`, fetches from the iterator until exhausted (i.e. until
          `StopIteration`), then returns.
        - If `Queue`, `.get()` from the `Queue` until `None` is received, then
          return.

        Args:
            stream: stream to consume.
            fps: video framerate.
            thread: whether to return immediately, and run in a separate thread
                instead of returning immediately.
        Raises:
            ValueError: data type/shape does not match channel specifications.
        """
        if isinstance(stream, Queue):
            stream = cast(Iterable[Data], Buffer(stream))
        if thread:
            Thread(
                target=self.consume, kwargs={"stream": stream, "fps": fps}
            ).start()
            return

        fourcc = self._cv2_module.VideoWriter_fourcc(*'MJPG')  # type: ignore
        cap = self._cv2_module.VideoWriter(
            self.path, fourcc, fps, (self.shape[1], self.shape[0]))
        for frame in stream:
            if not isinstance(frame, np.ndarray):
                raise ValueError("LzmaFrame does not allow raw data.")
            self._verify_type(frame)
            cap.write(frame)
        cap.release()
