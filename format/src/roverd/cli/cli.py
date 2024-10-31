"""CLI."""

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from typing import cast

from . import info, validate


def _cli_parser():
    commands = {"info": info, "validate": validate}
    parser = ArgumentParser(
        description="Rover Dataset CLI.",
        formatter_class=RawDescriptionHelpFormatter)

    subparsers = parser.add_subparsers()
    for name, command in commands.items():
        p = subparsers.add_parser(
            name, help=cast(str, command.__doc__).split('\n')[0],
            description=command.__doc__,
            formatter_class=RawDescriptionHelpFormatter)
        command._parse(p)
        p.set_defaults(_func=command._main)

    return parser


def _cli_main() -> None:
    """Dispatch scripts."""
    parser = _cli_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args._func(args)
