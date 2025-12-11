"""`roverd.channels` unit tests."""

import numpy as np
import pytest
from roverd.channels import (
    LzmaChannel,
    LzmaFrameChannel,
    RawChannel,
    VideoChannel,
)


def _create_random_data(size: tuple[int, ...]) -> np.ndarray:
    """Create random data of the given size."""
    return np.random.default_rng(42).random(size)


@pytest.mark.parametrize("ctype", [LzmaFrameChannel, LzmaChannel, RawChannel])
@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_raw(ctype, dtype, tmp_path):
    """Test raw binary channels."""
    shape = (4, 5)
    data = (_create_random_data((3, *shape)) * 1000).astype(dtype)

    ctype(str(tmp_path / "write"), shape=shape, dtype=data.dtype).write(data)
    ctype(str(tmp_path / "consume"), shape=shape, dtype=data.dtype).consume(
        frame[None] for frame in data)

    for path in ["write", "consume"]:
        channel = ctype(str(tmp_path / path), shape=shape, dtype=data.dtype)
        np.testing.assert_array_equal(channel.read(), data)

        # Test reading via .stream
        streamed_data = np.array(list(channel.stream(transform=lambda x: x * 2)))
        np.testing.assert_array_equal(streamed_data, data * 2)


def test_video_channel(tmp_path):
    """Test VideoChannel for write, read, consume, and stream."""
    shape = (3, 8, 16, 3)  # Example shape for video frames
    data = (_create_random_data(shape) * 255).astype(np.uint8)

    VideoChannel(str(tmp_path / "write.avi"), shape=shape[1:], dtype=data.dtype).write(data)
    VideoChannel(str(tmp_path / "consume.avi"), shape=shape[1:], dtype=data.dtype).consume(
        frame for frame in data)

    for path in ["write.avi", "consume.avi"]:
        channel = VideoChannel(str(tmp_path / path), shape=shape[1:], dtype=data.dtype)
        assert channel.shape == data.shape[1:]

        # Test reading via .stream
        streamed_data = np.array(list(channel.stream()))
        assert streamed_data.shape == data.shape


def test_video_channel_resolution(tmp_path):
    """Test VideoChannel with resolution downsampling."""
    shape = (3, 64, 128, 3)  # Original shape: 3 frames, 64x128 resolution
    data = (_create_random_data(shape) * 255).astype(np.uint8)

    # Write at full resolution
    VideoChannel(
        str(tmp_path / "full.avi"), shape=shape[1:], dtype=data.dtype
    ).write(data)

    # Read with different resolutions
    resolutions = [
        ((64, 32), (3, 32, 64, 3)),  # Half size: (width, height) -> (samples, height, width, channels)
        ((32, 16), (3, 16, 32, 3)),  # Quarter size
    ]

    for resolution, expected_shape in resolutions:
        channel = VideoChannel(
            str(tmp_path / "full.avi"),
            shape=shape[1:],
            dtype=data.dtype,
            resolution=resolution
        )

        # Test via .read()
        read_data = channel.read()
        assert read_data.shape == expected_shape, \
            f"Expected {expected_shape}, got {read_data.shape} for resolution {resolution}"

        # Test via .stream()
        streamed_data = np.array(list(channel.stream()))
        assert streamed_data.shape == expected_shape


@pytest.mark.parametrize("interp", ["nearest", "linear", "cubic", "area", "lanczos"])
def test_video_channel_interpolation(tmp_path, interp):
    """Test VideoChannel with different interpolation methods."""
    shape = (2, 64, 64, 3)
    data = (_create_random_data(shape) * 255).astype(np.uint8)

    VideoChannel(
        str(tmp_path / "test.avi"), shape=shape[1:], dtype=data.dtype
    ).write(data)

    # Read with downsampling using specified interpolation
    channel = VideoChannel(
        str(tmp_path / "test.avi"),
        shape=shape[1:],
        dtype=data.dtype,
        resolution=(32, 32),
        interp=interp
    )

    read_data = channel.read()
    assert read_data.shape == (2, 32, 32, 3)


@pytest.mark.parametrize("ctype", [LzmaFrameChannel, RawChannel, VideoChannel])
def test_random_reading(ctype, tmp_path):
    """Test random reading for channels."""
    shape = (5, 8, 16, 3)  # Example shape for multiple frames
    dtype = np.uint8 if ctype is VideoChannel else np.float32
    data = (_create_random_data(shape) * (255 if dtype == np.uint8 else 1000)).astype(dtype)

    channel = ctype(str(tmp_path / "random.avi"), shape=shape[1:], dtype=data.dtype)
    channel.write(data)

    # Test random access
    for i in [0, 3]:  # Example random indices
        if ctype is VideoChannel:
            assert channel[i].shape == data[i][None].shape
        else:
            np.testing.assert_array_equal(channel[i], data[i][None])


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_rawchannel_memmap(dtype, tmp_path):
    """Test memmap functionality in RawChannel."""
    shape = (10, 4, 5)  # Example shape for data
    data = (_create_random_data(shape) * 1000).astype(dtype)

    # Write data
    channel = RawChannel(
        str(tmp_path / "memmap"), shape=shape[1:], dtype=data.dtype)
    channel.write(data)

    # Read data back and verify
    mmap = channel.memmap()
    np.testing.assert_array_equal(mmap, data)
