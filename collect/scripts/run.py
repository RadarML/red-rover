"""Helper for launch data collection processes."""

import os
import yaml
import json
import logging
import traceback

import rover


class JsonFormatter(logging.Formatter):
    """Print log entries as json; each line corresponds to a single entry."""

    def format(self, record):
        return json.dumps({
            'ts': self.formatTime(record),
            'lvl': record.levelname,
            'mod': record.name,
            'msg': record.getMessage()
        })


def _parse(p):
    p.add_argument(
        "-c", "--config", default=os.getenv('ROVER_CFG'), help="Config file.")
    p.add_argument(
        "-l", "--log_level", type=int, default=logging.INFO,
        help="Logging level (defaut=info=20; 0-50).")
    p.add_argument("-s", "--sensor", default="radar", help="Sensor to launch.")


def _main(args):
    root = logging.getLogger()
    root.setLevel(args.log_level)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    path = os.path.abspath(os.path.expanduser(args.config))
    with open(path) as f:
        try:
            cfg = yaml.load(f, Loader=yaml.FullLoader)[args.sensor]
        except KeyError:
            root.critical("Sensor '{}' not defined in {}.".format(
                args.sensor, args.config))
            exit(-1)

    try:
        rover.SENSORS[cfg["type"]](name=args.sensor, **cfg["args"]).loop()
    except Exception as e:
        root.critical("".join(traceback.format_exception(e)))
    except KeyboardInterrupt:
        root.critical("Terminating due to KeyboardInterrupt.")
        exit(0)
