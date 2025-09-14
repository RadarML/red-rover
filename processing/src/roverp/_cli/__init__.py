"""Roverp CLI entry point."""

from typing import Annotated, Any, Union

import tyro

from .anonymize import cli_anonymize
from .report import cli_report
from .segment import cli_segment
from .sensorpose import cli_sensorpose


def make_annotation(name, func):
    return Annotated[
        Any,
        tyro.conf.subcommand(
            name, description=func.__doc__.split('\n\n')[0],
            constructor=func
        )
    ]


def cli_main() -> None:
    commands = {
        "anonymize": cli_anonymize,
        "report": cli_report,
        "sensorpose": cli_sensorpose,
        "segment": cli_segment,
    }

    return tyro.cli(Union[  # type: ignore
        tuple(make_annotation(k, commands[k]) for k in sorted(commands.keys()))
    ])
