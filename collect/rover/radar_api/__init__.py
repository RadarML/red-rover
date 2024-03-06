r"""AWR1843Boost Radar & DCA1000EVM Capture Card API.

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    TI AWR1843Boost/DCA1000EVM Raw I/Q API

References
----------
[1] DCA1000EVM Data Capture Card User's Guide (Rev A)
    https://www.ti.com/lit/ug/spruij4a/spruij4a.pdf?ts=1709104212742
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
[7] MMwave Radar Device ADC Raw Capture Data
    https://www.ti.com/lit/an/swra581b/swra581b.pdf?ts=1609161628089
"""

from . import dca_types, awr_types
from .awr_api import AWR1843
from .dca_api import DCA1000EVM
from .system import AWRSystem
from .config import RadarConfig, CaptureConfig

__all__ = [
    "AWRSystem", "RadarConfig", "CaptureConfig",
    "awr_types", "dca_types", "AWR1843", "DCA1000EVM"]
