"""

References
----------
[1] DCA1000EVM Data Capture Card User's Guide (Rev A)
[2] SourceCode folder in `mmWaveStudio/ReferenceCode/DCA1000`; some exerpts
    are provided in this repository.
"""

import sys
import socket
import logging
import struct

from beartype.typing import NamedTuple, Optional
import dca_types as types


class CommandRequest(NamedTuple):
    """Command request protocol."""

    cmd: types.Command
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


class CommandResponse(NamedTuple):
    """Command response protocol."""

    cmd: int
    status: int

    @classmethod
    def from_bytes(cls, packet: bytes) -> "CommandResponse":
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


class DCA1000EVM:
    """DCA1000EVM Interface.
    
    Documented by [1]; based on a UDP protocol (Sec 5, [1]). Included C++
    source code exerpts from the mmWave API are used as a secondary reference.
    """

    _MAX_PACKET_SIZE = 4096

    def _create_socket(
        self, addr: tuple[str, int], timeout: float
    ) -> socket.socket:
        """Create socket."""
        sock = socket.socket(
            socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.bind(addr)
        sock.settimeout(timeout)
        self.log.info("Connected to {}:{}".format(*addr))
        return sock

    def __init__(
        self, sys_ip: str = "192.168.33.30", fpga_ip: str = "192.168.33.180",
        record_port: int = 4098, config_port: int = 4096, timeout: float = 1.0,
        name: str = "DCA1000EVM"
    ) -> None:
        self.log = logging.getLogger(name=name)

        self.sys_ip = sys_ip
        self.fpga_ip = fpga_ip
        self.config_port = config_port
        self.record_port = record_port

        self.config_socket = self._create_socket((sys_ip, config_port), timeout)
        self.data_socket = self._create_socket((sys_ip, record_port), timeout)

    def system_aliveness(self) -> None:
        """Simple ping to query system status."""
        cmd = CommandRequest(types.Command.SYSTEM_ALIVENESS, bytes())
        self.config_request(cmd, desc="Verify FPGA connectivity")

    # untested
    def reset_ar_device(self) -> CommandRequest:
        """Reset AR device; it's not clear what an 'AR Device' is."""
        return CommandRequest(types.Command.RESET_AR_DEV, bytes())

    def config_fpga(
        self, log: types.Log, lvds: types.LVDS, transfer: types.DataTransfer,
        capture: types.DataCapture, format: types.DataFormat
    ) -> None:
        """Configure FPGA.
        
        NOTE: This seems to cause the FPGA to ignore requests for a short time
        after. Sending `system_aliveness` pings until it responds seems to be
        the best way to check when it's ready again.
        """
        self.log.info("Configuring FPGA: {}, {}, {}, {}, {}".format(
            log, lvds, transfer, capture, format))
        cfg = struct.pack(
            "HHHHHH", log.value, lvds.value, transfer.value,
            capture.value, format.value, types.FPGA_CONFIG_DEFAULT_TIMER)
        cmd = CommandRequest(types.Command.CONFIG_FPGA, cfg)
        self.config_request(cmd, desc="Configure FPGA")

        self.log.info("Testing/waiting for FPGA to respond to new requests.")
        for _ in range(30):
            try:
                return self.system_aliveness()
            except TimeoutError:
                pass
        else:
            self.log.error(
                "FPGA stopped responding to requests after configuring.")

    def config_eeprom(
        self, sys_ip: str = "192.168.33.30", fpga_ip: str = "192.168.33.180",
        fpga_mac: str = "12:34:56:78:90:12",
        config_port: int = 4096, record_port: int = 4098
    ) -> None:
        """Configure EEPROM; contains IP, MAC, port information.
        
        NOTE: Use with extreme caution. This should never be used in normal
        operation. May require delay before use depending on the previous cmd.
        """
        cfg = struct.pack(
            "B" * (4 + 4 + 6) + "HH",
            *types.ipv4_to_int(sys_ip), *types.ipv4_to_int(fpga_ip),
            *types.mac_to_int(fpga_mac), config_port, record_port)

        cmd = CommandRequest(types.Command.CONFIG_EEPROM, cfg)
        self.config_request(cmd, desc="Configure EEPROM")

    # untested
    def start_record(self) -> None:
        """Start recording data."""
        cmd = CommandRequest(types.Command.START_RECORD, bytes())
        self.config_request(cmd, desc="Start recording")

    # untested
    def stop_record(self) -> None:
        """Stop recording data."""
        cmd = CommandRequest(types.Command.STOP_RECORD, bytes())
        self.config_request(cmd, desc="Stop recording")

    def read_fpga_version(self) -> tuple[int, int, bool]:
        """Get current FPGA version."""
        cmd = CommandRequest(types.Command.READ_FPGA_VERSION, bytes())
        resp = self.config_request(cmd)
        if resp.status < 0:
            self.log.error(
                "Unable to read FPGA version: {}".format(resp.status))
            return (0, 0, False)
        else:
            major = resp.status & 0x7F
            minor = 0x7f & (resp.status >> 7)
            playback = resp.status & 0x4000
            self.log.info(
                "FPGA Version: {}.{} [mode={}]".format(
                    major, minor, "playback" if playback else "record"))
            return (major, minor, playback != 0)

    def config_record(self, delay: float = 25.0) -> None:
        """Configure data packets (with a packet delay in us)."""
        converted = int(
            delay * types.FPGA_CLK_CONVERSION_FACTOR
            / types.FPGA_CLK_PERIOD_IN_NANO_SEC)
        cfg = struct.pack("HHH", types.MAX_BYTES_PER_PACKET, converted, 0)
        cmd = CommandRequest(types.Command.CONFIG_RECORD, cfg)
        self.config_request(cmd, desc="Configure recording")

    def config_request(
        self, cmd: CommandRequest, desc: Optional[str] = None
    ) -> CommandResponse:
        """Send config command."""
        payload = cmd.to_bytes()
        self.config_socket.sendto(payload, (self.fpga_ip, self.config_port))
        self.log.debug("Sent: {}".format(cmd))

        raw, _ = self.config_socket.recvfrom(self._MAX_PACKET_SIZE)
        response = CommandResponse.from_bytes(raw)
        self.log.debug("Received: {}".format(response))

        if desc is not None:
            if response.status == 0:
                self.log.info("Success: {}".format(desc))
            else:
                self.log.error("Failure: {} (status={})".format(
                    desc, response.status))
        return response
            

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
dca = DCA1000EVM()
dca.system_aliveness()
dca.read_fpga_version()
dca.config_fpga(
    log=types.Log.RAW_MODE, lvds=types.LVDS.FOUR_LANE,
    transfer=types.DataTransfer.CAPTURE, format=types.DataFormat.BIT16,
    capture=types.DataCapture.ETH_STREAM)
dca.config_record(delay=25)
