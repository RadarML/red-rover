"""Test cases for trace/dataset reading and splitting."""

import json
import os
from functools import partial

import pytest
from abstract_dataloader import generic
from roverd import Dataset, Trace, sensors, types


@pytest.fixture
def test_trace_path():
    """Get the path to test trace from environment variable."""
    trace_path = os.environ.get("ROVERD_TEST_TRACE")
    if not trace_path:
        pytest.skip("ROVERD_TEST_TRACE environment variable not set")
    return trace_path


@pytest.mark.parametrize("smooth", [True, False])
def test_dataset_api(test_trace_path, smooth):
    """Test Dataset API with typed sensors as shown in roverd documentation."""
    def _check_shape(sample, attr, sensor, channel):
        with open(os.path.join(test_trace_path, sensor, "meta.json")) as f:
            meta = json.load(f)
        expected = tuple(meta[channel]['shape'])
        actual = getattr(sample, attr).shape[2:]

        assert actual == expected

    if smooth:
        spec = {
            "radar": partial(sensors.XWRRadar, correction="auto"),
            "lidar": partial(sensors.OSLidarDepth, correction="auto"),
            "camera": partial(sensors.Camera, correction="auto"),
            "imu": partial(sensors.IMU, correction="auto"),}
    else:
        spec = {
            "radar": "auto", "lidar": "auto",
            "camera": "auto", "imu": "auto"}

    dataset = Dataset.from_config(
        [test_trace_path], sync=generic.Nearest("lidar"),
        sensors=spec)

    assert len(dataset.traces) > 0
    sample = dataset[3]

    assert isinstance(sample["radar"], types.XWRRadarIQ)
    for key, ch in [("iq", "iq"), ("timestamps", "ts"), ("valid", "valid")]:
        _check_shape(sample["radar"], key, "radar", ch)

    assert isinstance(sample["lidar"], types.OSDepth)
    for key, ch in [("rng", "rng"), ("timestamps", "ts")]:
        _check_shape(sample["lidar"], key, "lidar", ch)

    assert isinstance(sample["camera"], types.CameraData)
    for key, ch in [("image", "video.avi"), ("timestamps", "ts")]:
        _check_shape(sample["camera"], key, "camera", ch)

    assert isinstance(sample["imu"], types.IMUData)
    for key, ch in [("acc", "acc"), ("rot", "rot"), ("avel", "avel"), ("timestamps", "ts")]:
        _check_shape(sample["imu"], key, "imu", ch)


def test_oslidar(test_trace_path):
    """Test reading OSLidar data."""
    def _check_shape(sample, attr, sensor, channel):
        with open(os.path.join(test_trace_path, sensor, "meta.json")) as f:
            meta = json.load(f)
        expected = tuple(meta[channel]['shape'])
        actual = getattr(sample, attr).shape[2:]

        assert actual == expected

    trace = Trace.from_config(
        test_trace_path, sync=generic.Nearest("lidar"), sensors={"lidar": sensors.OSLidar})
    sample = trace[3]
    assert isinstance(sample["lidar"], types.OSData)
    for key, ch in [("rng", "rng"), ("nir", "nir"), ("rfl", "rfl"), ("timestamps", "ts")]:
        _check_shape(sample["lidar"], key, "lidar", ch)


def test_camera_resolution(test_trace_path):
    """Test reading camera data with resolution control."""
    # Read original resolution
    trace_full = Trace.from_config(
        test_trace_path, sync=generic.Nearest("camera"),
        sensors={"camera": sensors.Camera})
    sample_full = trace_full[3]
    assert isinstance(sample_full["camera"], types.CameraData)
    original_shape = sample_full["camera"].image.shape[2:]  # (height, width, channels)

    # Read with custom resolution (half size)
    target_resolution = (original_shape[1] // 2, original_shape[0] // 2)  # (width, height)
    trace_resized = Trace.from_config(
        test_trace_path, sync=generic.Nearest("camera"),
        sensors={"camera": partial(sensors.Camera, resolution=target_resolution)})
    sample_resized = trace_resized[3]
    assert isinstance(sample_resized["camera"], types.CameraData)

    # Check that resolution was applied: (batch, time, height, width, channels)
    expected_shape = (target_resolution[1], target_resolution[0], original_shape[2])
    actual_shape = sample_resized["camera"].image.shape[2:]
    assert actual_shape == expected_shape, \
        f"Expected shape {expected_shape}, got {actual_shape}"


def test_trace_generic(test_trace_path):
    """Test reading a trace with generic sensors."""
    def _check_shape(sample, sensor, channel):
        with open(os.path.join(test_trace_path, sensor, "meta.json")) as f:
            meta = json.load(f)
        expected = tuple(meta[channel]['shape'])
        actual = sample[channel].shape[1:]

        assert actual == expected

    trace = Trace.from_config(test_trace_path, sync=generic.Nearest("lidar"))
    sample = trace[3]

    for key, ch in [("iq", "iq"), ("timestamps", "ts"), ("valid", "valid")]:
        _check_shape(sample["radar"], "radar", ch)
    for key, ch in [("rng", "rng"), ("timestamps", "ts")]:
        _check_shape(sample["lidar"], "lidar", ch)
    for key, ch in [("image", "video.avi"), ("timestamps", "ts")]:
        _check_shape(sample["camera"], "camera", ch)
    for key, ch in [("acc", "acc"), ("rot", "rot"), ("avel", "avel"), ("timestamps", "ts")]:
        _check_shape(sample["imu"], "imu", ch)
