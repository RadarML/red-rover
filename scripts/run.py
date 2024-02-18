"""Helper for launch data collection processes."""

import os
import yaml
import subprocess


def _parse(p):
    p.add_argument(
        "-c", "--config", default="config.yaml", help="Config file.")
    p.add_argument("-s", "--sensor", default="radar", help="Sensor to launch.")


def _main(args):
    path = os.path.abspath(os.path.expanduser(args.config))
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if args.sensor not in cfg:
        print("Sensor '{}' not defined.".format(args.sensor))
        exit(-1)

    os.chdir("./rover2")
    python = "./env/bin/python"
    subprocess.call([python, cfg[args.sensor]["script"], args.sensor, path])
