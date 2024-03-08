"""AWR1843 API Defines [4]."""

from enum import Enum


class AWRException(Exception):
    """Error raised by the Radar (via non-normal return message)."""
    pass


class LVDSFormat(Enum):
    """LVDS data format."""

    DISABLED = 0
    ADC = 1
    _RESERVED2 = 2
    _RESERVED3 = 3
    CP_ADC_CQ = 4


class DFEMode(Enum):
    """Frame type."""

    LEGACY = 1
    # CONTINUOUS = 2 (not supported)
    ADVANCED = 3


class ADCDepth(Enum):
    """ADC bith depth."""

    BIT12 = 0
    BIT14 = 1
    BIT16 = 2


class ADCFormat(Enum):
    """ADC output format.
    
    COMPLEX_1X has the image band filtered out, while COMPLEX_2X does not.
    """

    REAL = 0
    COMPLEX_1X = 1
    COMPLEX_2X = 2


class SampleSwap(Enum):
    """ADC I/Q bit order.
    
    NOTE: MSB_LSB_QI doesn't seem to work.
    """

    MSB_LSB_QI = 0
    MSB_LSB_IQ = 1


class HPFCornerFreq1(Enum):
    """High pass filter 1 corner frequency."""

    KHZ175 = 0
    KHZ235 = 1
    KHZ350 = 2
    KHZ700 = 3


class HPFCornerFreq2(Enum):
    """High pass filter 1 corner frequency."""

    KHZ350 = 0
    KHZ700 = 1
    MHZ1_4 = 2
    MHZ2_8 = 3
