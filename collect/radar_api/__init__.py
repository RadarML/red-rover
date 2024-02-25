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
[4] mmWave SDK user guide, Table 1 (Page 19)
    https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-PIrUeCYr3X/03.06.00.00-LTS/mmwave_sdk_user_guide.pdf
[5] mmWave Studio
    https://www.ti.com/tool/MMWAVE-STUDIO
[6] AWR1843 Data Sheet
    https://www.ti.com/lit/ds/symlink/awr1843.pdf?ts=1708800208074
"""

from . import dca_types
from .awr_api import AWR1843
from .dca_api import DCA1000EVM
from .dca_writer import RadarDataWriter

__all__ = [
    "dca_types", "AWR1843", "DCA1000EVM", "RadarDataWriter"
]
