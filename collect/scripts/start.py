"""Start data collection."""

import os
from datetime import datetime

from .control import Controller


def _parse(p):
    p.add_argument(
        "-c", "--config", default="config.yaml", help="Config file.")
    p.add_argument("-p", "--path", default="./data", help="Dataset directory.")


def _main(args):
    ctrl = Controller.from_config(args.config)

    dt = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    ctrl.start(os.path.join(os.path.abspath(args.path), dt))
