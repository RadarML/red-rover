"""Roverd CLI entry point."""

import tyro

from .info import cli_info
from .validate import cli_validate


def cli_main() -> None:
    tyro.extras.subcommand_cli_from_dict({
        "info": cli_info,
        "validate": cli_validate,
    })
