"""Get configuration key."""

import os

import yaml


def _parse(p):
    p.add_argument(
        "-c", "--config", default=os.getenv('ROVER_CFG'), help="Config file.")
    p.add_argument(
        "-p", "--path", default='', help="Configuration key path to get.")


def _main(args):
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    try:
        for p in args.path.split('/'):
            cfg = cfg[p]
        print(cfg)
    except KeyError:
        print("Unknown key:", args.path)
        exit(-1)
