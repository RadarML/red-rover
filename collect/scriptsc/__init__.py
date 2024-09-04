r"""Rover data collection platform.
::
     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform
.
"""  # noqa: D205

from .control import Controller

_scripts = sorted(["run", "start", "stop", "exit", "get_config", "info"])
__all__ = ["Controller"]
