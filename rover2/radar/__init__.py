r"""AWR1843Boost Radar & DCA1000EVM Capture Card API.

     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    TI AWR1843Boost/DCA1000EVM Raw I/Q API

References
----------
[1] DCA1000EVM Data Capture Card User's Guide (Rev A)
[2] SourceCode folder in `mmWaveStudio/ReferenceCode/DCA1000`; some exerpts
    are provided in this repository.
"""

from . import dca_types
from .dca_api import DCA1000EVM
from .dca_writer import RadarDataWriter

__all__ = [
    "dca_types", "DCA1000EVM", "RadarDataWriter"
]
