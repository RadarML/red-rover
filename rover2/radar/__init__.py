r"""AWR1843Boost Radar & DCA1000EVM Capture Card API.

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    TI AWR1843Boost/DCA1000EVM Raw I/Q API

References
----------
[1] DCA1000EVM Data Capture Card User's Guide (Rev A)
[2] `ReferenceCode/DCA1000/SourceCode` folder in the mmWave Studio install
    directory; relevant exerpts are included in `reference/mmWave_API`.
[3] `packages/ti/demo/xwr18xx/mmw` folder in the mmWave SDK install directory;
    relevant exerpts are included in `reference/demo_xwr18xx`.
"""

from . import dca_types
from .dca_api import DCA1000EVM
from .dca_writer import RadarDataWriter

__all__ = [
    "dca_types", "DCA1000EVM", "RadarDataWriter"
]
