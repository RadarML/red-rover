"""Close data collection processes and clean up."""

from .control import Controller


def _parse(p):
    p.add_argument(
        "-c", "--config", default="config.yaml", help="Config file.")


def _main(args):
    Controller.from_config(args.config).exit()
