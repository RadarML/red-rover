"""Validate dataset files."""

import os
import warnings
from typing import Sequence, cast

import yaml

from roverd import Dataset, Trace, sensors


def _validate_schema(path: str, schema) -> list[str]:
    ds = Trace.from_config(path, include_virtual=True)
    errors = []

    # Check schema
    for sensor, channels in schema.items():
        if sensor == "_":
            for file in channels:
                if not os.path.exists(os.path.join(path, file)):
                    errors.append(f"Missing file: {file}")
        elif sensor not in ds.sensors:
            errors.append(f"Missing sensor: {sensor}")
        else:
            sns = ds[sensor]
            for channel in channels:
                try:
                    fs = sns[channel].filesize
                    if fs == 0:
                        errors.append(f"Bad channel: {sensor}/{channel}")
                except (FileNotFoundError, KeyError):
                    errors.append(f"Missing channel: {sensor}/{channel}")
                except Exception as e:
                    errors.append(
                        f"Caused other error: {sensor}/{channel}")
                    errors.append(f"    (error): {repr(e)}")

    return errors


def _validate_consistency(path: str, fix_errors: bool = False) -> list[str]:
    ds = Trace.from_config(path, include_virtual=True)
    errors = []

    # Check consistency
    all_sensors = [s for s in os.listdir(path) if s in ds.sensors]
    for sensor_name in all_sensors:
        sensor = cast(sensors.DynamicSensor, ds[sensor_name])
        for channel in sensor.channels:
            try:
                fs = sensor[channel].filesize
                if fs == 0:
                    errors.append(f"Bad channel: {sensor}/{channel}")
            except FileNotFoundError:
                if fix_errors:
                    del sensor.config[channel]
                    sensor._flush_config()
                    errors.append(f"Removed metadata: {sensor}/{channel}")
                else:
                    errors.append(f"No data: {sensor}/{channel}")

    return errors


def cli_validate(
    path: Sequence[str], /, schema: str | None = None,
    fix_errors: bool = False
) -> None:
    """Validate dataset files.

    ```sh
    $ roverd validate /data/grt
    Validate: 166 traces with 0 containing errors.
    $ echo $?
    0  # would be 1 if there were any errors
    ```

    ??? info "Usage with custom schema"

        ```yaml title="grt.yaml"
        camera: ["ts", "video.avi"]
        _camera: ["ts", "segment"]
        lidar: ["ts", "rng"]
        radar: ["ts", "iq"]
        imu: ["ts", "rot", "acc", "avel"]
        _:
        - _camera/pose.npz
        - _fusion/indices.npz
        - _lidar/pose.npz
        - _radar/pose.npz
        - _slam/trajectory.csv
        ```

        ```sh title="/bin/sh"
        $ roverd validate /data/grt --schema grt.yaml
        Validate: 166 traces with 0 containing errors.
        ```

    Args:
        path: Target path or list of paths to validate.
        schema: Dataset file schema (yaml) to check. If not specified, uses a
            default schema which corresponds to raw files which are expected
            to be collected by the `red-rover` rig.
        fix_errors: If `True`, fix consistency errors (data not present, but
            metadata is present).
    """
    # We'll get a lot of warings for missing timestamps. Don't warn, since
    # the schema will explicitly catch them if the user cares.
    warnings.filterwarnings(
        "ignore", message="Sensor metadata does not contain 'ts' channel")

    if schema is None:
        _schema = {
            "lidar": ["ts", "rfl", "nir", "rng"],
            "radar": ["ts", "iq", "valid"],
            "camera": ["ts", "video.avi"],
            "imu": ["ts", "rot", "acc", "avel"]
        }
    else:
        with open(schema) as f:
            _schema = yaml.load(f, Loader=yaml.SafeLoader)

    datasets = Dataset.find_traces(*path)
    n_errors = 0
    for path in datasets:
        errors = _validate_schema(path, _schema)
        errors += _validate_consistency(path, fix_errors=fix_errors)
        if errors:
            n_errors += 1
            print(path)
            print('\n'.join("    " + x for x in errors))

    if n_errors > 0:
        print("")
    print(
        f"Validate: {len(datasets)} traces with {n_errors} containing "
        "errors.")
    if n_errors > 0:
        exit(1)
