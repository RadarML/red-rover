"""Helper for launch data collection processes."""

import os
import yaml
import logging

import rover


def _parse(p):
    p.add_argument(
        "-c", "--config", default="config.yaml", help="Config file.")
    p.add_argument(
        "-l", "--log_level", type=int, default=logging.INFO,
        help="Logging level (defaut=info=20; 0-50).")
    p.add_argument("-s", "--sensor", default="radar", help="Sensor to launch.")


def _main(args):
    path = os.path.abspath(os.path.expanduser(args.config))
    with open(path) as f:
        try:
            cfg = yaml.load(f, Loader=yaml.FullLoader)[args.sensor]
        except KeyError:
            print("Sensor '{}' not defined in {}.".format(
                args.sensor, args.config))
            exit(-1)

    logging.basicConfig(level=args.log_level)
    rover.SENSORS[cfg["type"]](name=args.sensor, **cfg["args"]).loop()
