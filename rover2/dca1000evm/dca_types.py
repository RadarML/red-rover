"""DCA1000EVM API Defines."""

from enum import Enum


class Command(Enum):
    """Command request codes; see `rf_api.h:CMD_CODE_*` [2]."""

    RESET_FPGA = 0x01
    RESET_AR_DEV = 0x02
    CONFIG_FPGA = 0x03
    CONFIG_EEPROM = 0x04
    START_RECORD = 0x05
    STOP_RECORD = 0x06
    START_PLAYBACK = 0x07
    STOP_PLAYBACK = 0x08
    SYSTEM_ALIVENESS = 0x09
    ASYNC_STATUS = 0x0A
    CONFIG_RECORD = 0x0B
    CONFIG_AR_DEV = 0x0C
    INIT_FPGA_PLAYBACK = 0x0D
    READ_FPGA_VERSION = 0x0E


class Log(Enum):
    """Data log mode; see `rf_api.h:enum CONFIG_LOG_MODE` [2]."""

    RAW_MODE = 1
    MULTI_MODE = 2


class LVDS(Enum):
    """LVDS mode (number of lanes); see `rf_api.h:enum CONFIG_LVDS_MODE` [2].
    
    TI Notes:
    - AR1243 - 4 lane
    - AR1642 - 2 lane
    """

    FOUR_LANE = 1
    TWO_LANE = 2


class DataTransfer(Enum):
    """Data transfer mode; see `rf_api.h:enum CONFIG_TRANSFER_MODE` [2]."""

    CAPTURE = 1
    PLAYBACK = 2


class DataFormat(Enum):
    """Data format (bit depth); see `rf_api.h:enum CONFIG_FORMAT_MODE` [2]."""

    BIT12 = 1
    BIT14 = 2
    BIT16 = 3


class DataCapture(Enum):
    """Data capture mode; see `rf_api.h:enum CONFIG_CAPTURE_MODE` [2]."""

    SD_STORAGE = 1
    ETH_STREAM = 2


FPGA_CONFIG_DEFAULT_TIMER = 30
"""LVDS timeout is always 30 (units not documented / unknown)."""


class Status:
    """Status codes."""

    SUCCESS = 0
    FAILURE = 1
