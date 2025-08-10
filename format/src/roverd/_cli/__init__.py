"""Roverd CLI entry point."""

from typing import Annotated, Any, Union

import tyro

from .extract import cli_extract
from .info import cli_info
from .rosbag import cli_rosbag
from .validate import cli_validate


def make_annotation(name, func):
    return Annotated[
        Any,
        tyro.conf.subcommand(
            name, description=func.__doc__.split('\n')[0],
            constructor=func
        )
    ]


def cli_main() -> None:
    commands = {
        "extract": cli_extract,
        "info": cli_info,
        "validate": cli_validate,
        "rosbag": cli_rosbag,
    }

    return tyro.cli(Union[  # type: ignore
        tuple(make_annotation(k, commands[k]) for k in sorted(commands.keys()))
    ])
