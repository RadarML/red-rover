"""Start data collection."""

import os
from datetime import datetime

from .control import Controller


def _parse(p):
    p.add_argument(
        "-c", "--config", default=os.getenv('ROVER_CFG'), help="Config file.")
    p.add_argument("-p", "--path", default="./data", help="Dataset directory.")


def _main(args):
    ctrl = Controller.from_config(args.config)

    dt = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    path = os.path.join(os.path.abspath(args.path), dt)
    os.makedirs(path, exist_ok=True)

    with open(args.config) as f:
        contents = f.read()
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        f.write(contents)

    ctrl.start(path)
