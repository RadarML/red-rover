r"""Rover dataset file format python API.
::

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform
.
"""  # noqa: D205

from jaxtyping import install_import_hook
with install_import_hook("deepradar", "beartype.beartype"):

    from . import channels
    from . import sensors
    from .dataset import Dataset

__all__ = ["channels", "sensors", "Dataset"]
