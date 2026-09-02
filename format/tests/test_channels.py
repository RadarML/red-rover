"""`roverd.channels` unit tests."""

import errno

import numpy as np
import pytest
import roverd.channels.raw as raw_module
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


def test_rawchannel_retries_transient_eio(monkeypatch, tmp_path):
    """Retry a transient EIO and return the requested data."""
    data = np.arange(12, dtype=np.int16).reshape(3, 4)
    channel = RawChannel(str(tmp_path / "raw"), shape=(4,), dtype=data.dtype)
    channel.write(data)
    open_r = channel._open_r
    attempts = 0
    delays = []

    def flaky_open(path):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError(errno.EIO, "transient read failure", path)
        return open_r(path)

    monkeypatch.setattr(channel, "_open_r", flaky_open)
    monkeypatch.setattr(raw_module.time, "sleep", delays.append)

    np.testing.assert_array_equal(channel.read(), data)
    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_rawchannel_stops_after_bounded_eio_retries(monkeypatch, tmp_path):
    """Propagate EIO after the bounded retry budget is exhausted."""
    channel = RawChannel(str(tmp_path / "raw"), shape=(4,), dtype=np.int16)
    attempts = 0
    delays = []

    def failing_open(path):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EIO, "persistent read failure", path)

    monkeypatch.setattr(channel, "_open_r", failing_open)
    monkeypatch.setattr(raw_module, "_RAW_READ_ATTEMPTS", 3)
    monkeypatch.setattr(raw_module.time, "sleep", delays.append)

    with pytest.raises(OSError, match="persistent read failure"):
        channel.read(samples=1)
    assert attempts == 3
    assert delays == [0.25, 0.5]


def test_rawchannel_does_not_retry_other_oserrors(monkeypatch, tmp_path):
    """Propagate non-EIO operating-system errors immediately."""
    channel = RawChannel(str(tmp_path / "raw"), shape=(4,), dtype=np.int16)
    attempts = 0

    def denied_open(path):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EACCES, "permission denied", path)

    monkeypatch.setattr(channel, "_open_r", denied_open)

    with pytest.raises(OSError, match="permission denied"):
        channel.read(samples=1)
    assert attempts == 1


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
