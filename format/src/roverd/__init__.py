r"""Rover dataset file format python API.
::

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform

"""  # noqa: D205

from jaxtyping import install_import_hook

with install_import_hook("deepradar", "beartype.beartype"):
    from . import channels, sensors
    from .cli import _cli_main, _cli_parser
    from .dataset import Dataset

# We need to expose `_cli_parser` for sphinx.
__all__ = ["_cli_main", "_cli_parser", "channels", "sensors", "Dataset"]
