"""Close data collection processes and clean up."""

import os
from .control import Controller


def _parse(p):
    p.add_argument(
        "-c", "--config", default=os.getenv('ROVER_CFG'), help="Config file.")


def _main(args):
    Controller.from_config(args.config).exit()
