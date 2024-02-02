"""

References
----------
[1] DCA1000EVM Data Capture Card User's Guide (Rev A)
"""


from enum import Enum

import struct
from beartype.typing import NamedTuple


class SYS_ASYNC_STATUS(Enum):
    """System async status from FPGA."""
    NO_LVDS_DATA = 0
    NO_HEADER = 1
    EEPROM_FAILURE = 2
    SD_CARD_DETECTED = 3
    SD_CARD_REMOVED = 4
    SD_CARD_FULL = 5
    MODE_CONFIG_FAILURE = 6
    DDR_FULL = 7
    REC_COMPLETED = 8
    LVDS_BUFFER_FULL = 9
    PLAYBACK_COMPLETED = 10
    PLAYBACK_OUT_OF_SEQ = 11


class CommandRequest(NamedTuple):
    """Command request protocol.
    
    Little endian.
    
    Attributes
    ----------
    cmd: Command code.
    """

    cmd: int
    data: bytes

    def to_packet(self) -> bytes:
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
            0xa55a, self.command_code, len(self.data), self.data, 0xeeaa)


class CommandResponse(NamedTuple):
    """Command response protocol."""

    cmd: int
    status: int

    @classmethod
    def from_bytes(cls, packet: bytes) -> "CommandResponse":
        """Read packet."""
        header, command_code, status, footer = struct.unpack("HHHH")
        assert header == 0xa55a
        assert footer == 0xeeaa
        return cls(cmd=command_code, status=status)
