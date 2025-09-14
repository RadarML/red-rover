"""Blob channels consisting of folders of separate files."""

import os
from collections.abc import Sequence
from functools import cached_property

import numpy as np

from .abstract import BlobChannel


class NPZBlobChannel(BlobChannel):
    """Blob channel consisting of `.npz` files.

    Args:
        path: file path.
        dtype: data type, or string name of dtype (e.g. `u1`, `f4`).
        shape: data shape.
        workers: maximum number of worker threads to use for I/O.
        length: number of blobs, potentially calculated some more efficient
            way. If `None`, will be calculated by counting files in the
            directory.
        compress: whether to use compression when writing `.npz` files.

    Attributes:
        path: file path.
        type: numpy data type.
        shape: sample data shape.
        size: total file size, in bytes.
    """

    def __init__(
        self, path: str, dtype: str | type | np.dtype, shape: Sequence[int],
        workers: int = 8, length: int | None = None, compress: bool = False
    ) -> None:
        super().__init__(path, dtype, shape, workers, length)
        self.compress = compress

    def _read_blob(self, index: int) -> np.ndarray:
        """Load a blob from file.

        Args:
            index: index of the blob to load.

        Returns:
            The loaded blob as a numpy array.
        """
        filename = self._filename(index) + ".npz"
        if not os.path.exists(filename):
            raise IndexError(f"Blob index {index} does not exist.")

        with np.load(filename) as data:
            return data['data']

    def _write_blob(self, index: int, data: np.ndarray) -> None:
        """Write a blob to a file.

        Args:
            index: index of the blob to load.
            data: data to write.
        """
        filename = self._filename(index) + ".npz"
        if self.compress:
            np.savez_compressed(filename, data=data)
        else:
            np.savez(filename, data=data)


class JPEGBlobChannel(BlobChannel):
    """Blob channel consisting of `.jpg` files.

    Args:
        path: file path.
        dtype: data type, or string name of dtype (e.g. `u1`, `f4`).
        shape: data shape.
        workers: maximum number of worker threads to use for I/O.
        length: number of blobs, potentially calculated some more efficient
            way. If `None`, will be calculated by counting files in the
            directory.

    Attributes:
        path: file path.
        type: numpy data type.
        shape: sample data shape.
        size: total file size, in bytes.
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

    def _read_blob(self, index: int) -> np.ndarray:
        filename = self._filename(index) + ".jpg"
        if not os.path.exists(filename):
            raise IndexError(f"Blob index {index} does not exist.")

        img = self._cv2_module.imread(
            filename, self._cv2_module.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image at index {index}.")
        return self._cv2_module.cvtColor(img, self._cv2_module.COLOR_BGR2RGB)

    def _write_blob(self, index: int, data: np.ndarray) -> None:
        filename = self._filename(index) + ".jpg"
        img = self._cv2_module.cvtColor(data, self._cv2_module.COLOR_RGB2BGR)
        if not self._cv2_module.imwrite(filename, img):
            raise ValueError(f"Could not write image at index {index}.")
