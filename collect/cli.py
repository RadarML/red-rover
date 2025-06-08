"""Data collection CLI."""

import json
import logging
import os
import traceback
from datetime import datetime

import roverc
import yaml
from controller import Controller


def _get_config_path(config: str | None) -> str:
    if config is None:
        config = os.getenv('ROVER_CFG')
    if not config:
        raise ValueError("Config file must be specified or set in ROVER_CFG.")

    return config


def cli_exit(config: str | None = None) -> None:
    """Exit data collection processes and clean up.

    Args:
        config: path to the config file. If `None`, uses the `ROVER_CFG`
            environment variable.
    """
    Controller.from_config(_get_config_path(config)).exit()


def cli_get_config(path: str = '', config: str | None = None) -> None:
    """Get the value of a provided path from the target config.

    Args:
        path: '/'-separated path to the configuration key.
        config: path to the config file. If `None`, uses the `ROVER_CFG`
            environment variable.
    """
    with open(_get_config_path(config)) as f:
        cfg = yaml.safe_load(f)

    try:
        for p in path.split('/'):
            cfg = cfg[p]
        print(cfg)
    except KeyError:
        print("Unknown key:", path)
        exit(-1)


def cli_start(path: str = './data', config: str | None = None) -> None:
    """Start data collection.

    Args:
        path: dataset directory.
        config: path to the config file. If `None`, uses the `ROVER_CFG`
            environment variable.
    """
    config = _get_config_path(config)
    ctrl = Controller.from_config(config)

    dt = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    path = os.path.join(os.path.abspath(path), dt)
    os.makedirs(path, exist_ok=True)

    with open(config) as f:
        contents = f.read()
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        f.write(contents)

    ctrl.start(path)


def cli_stop(config: str | None = None) -> None:
    """Stop data collection.

    Args:
        config: path to the config file. If `None`, uses the `ROVER_CFG`
            environment variable.
    """
    Controller.from_config(_get_config_path(config)).stop()


class JsonFormatter(logging.Formatter):
    """Print log entries as json; each line corresponds to a single entry."""

    def format(self, record):
        return json.dumps({
            'ts': self.formatTime(record),
            'lvl': record.levelname,
            'mod': record.name,
            'msg': record.getMessage()
        })


def cli_run(
    config: str | None = None, sensor: str = "radar",
    log_level: int = logging.INFO
) -> None:
    """Run a sensor data collection process.

    Args:
        config: path to the config file. If `None`, uses the `ROVER_CFG`
            environment variable.
        sensor: name of the sensor to run.
        log_level: logging level (default=info=20; 0-50).
    """
    config = _get_config_path(config)

    root = logging.getLogger()
    root.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    path = os.path.abspath(os.path.expanduser(config))
    with open(path) as f:
        try:
            cfg = yaml.load(f, Loader=yaml.FullLoader)[sensor]
        except KeyError:
            root.critical("Sensor '{}' not defined in {}.".format(
                sensor, config))
            exit(-1)

    try:
        roverc.SENSORS[cfg["type"]](name=sensor, **cfg["args"]).loop()
    except Exception as e:
        root.critical("".join(traceback.format_exception(e)))
    except KeyboardInterrupt:
        root.critical("Terminating due to KeyboardInterrupt.")
        exit(0)


if __name__ == '__main__':
    import tyro

    tyro.extras.subcommand_cli_from_dict({
        "exit": cli_exit,
        "get-config": cli_get_config,
        "start": cli_start,
        "stop": cli_stop,
        "run": cli_run,
    })
