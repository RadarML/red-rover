"""DCA1000EVM API Defines [2]."""

import struct
from enum import Enum
from beartype.typing import cast, NamedTuple


class DCA100EVMException(Exception):
    """Error raised by the FPGA (via non-0 status)."""
    pass


class Command(Enum):
    """Command request codes; see `rf_api.h:CMD_CODE_*`."""

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
    """Data log mode; see `rf_api.h:enum CONFIG_LOG_MODE`."""

    RAW_MODE = 1
    MULTI_MODE = 2


class LVDS(Enum):
    """LVDS mode (number of lanes); see `rf_api.h:enum CONFIG_LVDS_MODE`.
    
    TI Notes:
    - AR1243 - 4 lane
    - AR1642 - 2 lane
    """

    FOUR_LANE = 1
    TWO_LANE = 2


class DataTransfer(Enum):
    """Data transfer mode; see `rf_api.h:enum CONFIG_TRANSFER_MODE`."""

    CAPTURE = 1
    PLAYBACK = 2


class DataFormat(Enum):
    """Data format (bit depth); see `rf_api.h:enum CONFIG_FORMAT_MODE`."""

    BIT12 = 1
    BIT14 = 2
    BIT16 = 3


class DataCapture(Enum):
    """Data capture mode; see `rf_api.h:enum CONFIG_CAPTURE_MODE`."""

    SD_STORAGE = 1
    ETH_STREAM = 2


FPGA_CONFIG_DEFAULT_TIMER = 30
"""LVDS timeout is always 30 (units not documented / unknown)."""

MAX_BYTES_PER_PACKET = 1470
"""Maximum number of bytes in a single FPGA data packet."""

FPGA_CLK_CONVERSION_FACTOR = 1000
"""Record packet delay clock conversion factor."""

FPGA_CLK_PERIOD_IN_NANO_SEC = 8
"""Record packet delay clock period in ns."""


class Status:
    """Status codes."""

    SUCCESS = 0
    FAILURE = 1


def ipv4_to_int(ipv4: str) -> tuple[int, int, int, int]:
    """Parse ipv4 string as a tuple of 4 integers."""
    addr = tuple(reversed(list(int(x) for x in ipv4.split('.'))))
    return cast(tuple[int, int, int, int], addr)


def mac_to_int(mac: str) -> tuple[int, int, int, int, int, int]:
    """Parse MAC address string as a tuple of 6 integers."""
    addr = tuple(reversed(list(int(x, 16) for x in mac.split(':'))))
    return cast(tuple[int, int, int, int, int, int], addr)


class Request(NamedTuple):
    """Command request protocol."""

    cmd: Command
    data: bytes

    def to_bytes(self) -> bytes:
        """Form into a single packet.
        
        < : assumed to be little endian. Not documented anywhere, but implied
            since mmWave API uses native linux/x86 structs, which are little
            endian.
        H : Header is always `0xA55A` (Table 13, [1]).
        H : Command code (Table 12, [1]).
        H : Data size; must be between 0 and 504 (Section 5.1, [1]).
        {}s : Payload; can be empty.
        H : Footer is always `0xEEAA` (Table 13, [1]).
        """
        assert len(self.data) < 504
        return struct.pack(
            "<HHH{}sH".format(len(self.data)),
            0xa55a, self.cmd.value, len(self.data), self.data, 0xeeaa)


class Response(NamedTuple):
    """Command response protocol."""

    cmd: int
    status: int

    @classmethod
    def from_bytes(cls, packet: bytes) -> "Response":
        """Read packet."""
        header, command_code, status, footer = struct.unpack("HHHH", packet)
        assert header == 0xa55a
        assert footer == 0xeeaa
        return cls(cmd=command_code, status=status)


class DataPacket(NamedTuple):
    """Data packet protocol."""

    sequence_number: int
    byte_count: int
    data: bytes

    @classmethod
    def from_bytes(cls, packet: bytes) -> "DataPacket":
        """Read packet.
        
        Packet format (Sec. 5.2, [1]):
        < : assumed to be little endian.
        L : 4-byte sequence number (packet number).
        Q : 6-byte byte count index; appended with x0000 to make a uint64.
        """
        sn, bc = struct.unpack('<LQ', packet[:10] + b'\x00\x00')
        return cls(sequence_number=sn, byte_count=bc, data=packet[10:])
