"""Validate dataset files."""

import os

import yaml

from ..dataset import Dataset


def _parse(p):
    p.add_argument(
        "-p", "--path", default=[], nargs='+',
        help="Target path or list of paths.")
    p.add_argument(
        "-s", "--schema", help="Dataset file schema (yaml) to check. "
        "If not specified, uses a default schema which corresponds to raw "
        "files which are expected to be collected by the Rover rig.")
    p.add_argument(
        "--fix_errors", action='store_true', default=False,
        help="Fix consistency errors (data not present, but metadata is "
        "present).")

def _validate_schema(path: str, schema) -> list[str]:
    ds = Dataset(path)
    errors = []

    # Check schema
    for sensor, channels in schema.items():
        if sensor == "_":
            for file in channels:
                if not os.path.exists(os.path.join(path, file)):
                    errors.append(f"Missing file: {file}")
        elif sensor not in ds:
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
    ds = Dataset(path)
    errors = []

    # Check consistency
    all_sensors = [s for s in os.listdir(path) if s in ds]
    for sensor_name in all_sensors:
        sensor = ds[sensor_name]
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


def _main(args):

    if args.schema is None:
        schema = Dataset.DEFAULT_SCHEMA
    else:
        with open(args.schema) as f:
            schema = yaml.load(f, Loader=yaml.SafeLoader)

    datasets = Dataset.find(*args.path)
    n_errors = 0
    for path in datasets:
        errors = _validate_schema(path, schema)
        errors += _validate_consistency(path, fix_errors=args.fix_errors)
        if errors:
            n_errors += 1
            print(path)
            print('\n'.join("    " + x for x in errors))

    if n_errors > 0:
        print("")
    print(
        f"Validate: {len(datasets)} datasets with {n_errors} containing "
        "errors.")
