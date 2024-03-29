"""Script dispatcher."""

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import importlib
from typing import cast


def dispatch(target: str) -> None:
    """Dispatch scripts to the specified module.
    
    The module should have a `_scripts` attribute with the list of valid
    scripts; the `__doc__` is used as the script description.
    """
    target_module = importlib.import_module(target)
    commands = {
        cmd: importlib.import_module("{}.{}".format(target, cmd))
        for cmd in target_module._scripts
    }
    parser = ArgumentParser(
        description=target_module.__doc__,
        formatter_class=RawDescriptionHelpFormatter)

    subparsers = parser.add_subparsers()
    for name, command in commands.items():
        p = subparsers.add_parser(
            name, help=cast(str, command.__doc__).split('\n')[0],
            description=command.__doc__,
            formatter_class=RawDescriptionHelpFormatter)
        command._parse(p)
        p.set_defaults(_func=command._main)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args._func(args)


if __name__ == '__main__':
    dispatch("scripts")
