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

from beartype.typing import NamedTuple
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
        data_port: int = 4098, config_port: int = 4096, timeout: float = 1.0,
        name: str = "DCA1000EVM"
    ) -> None:
        self.log = logging.getLogger(name=name)

        self.config_addr = (fpga_ip, config_port)
        self.response_addr = (sys_ip, config_port)
        self.data_addr = (sys_ip, data_port)
        self.config_socket = self._create_socket(self.response_addr, timeout)
        self.data_socket = self._create_socket(self.data_addr, timeout)

    def system_aliveness(self) -> None:
        """Simple ping to query system status."""
        cmd = CommandRequest(types.Command.SYSTEM_ALIVENESS, bytes())
        resp = self.config_request(cmd)
        if resp.status == types.Status.SUCCESS:
            self.log.info("FPGA system connectivity verified.")
        else:
            self.log.error("FPGA returned failure status.")

    # untested
    def reset_ar_device(self) -> CommandRequest:
        """Reset AR device; it's not clear what an 'AR Device' is."""
        return CommandRequest(types.Command.RESET_AR_DEV, bytes())

    def config_fpga(
        self, log: types.Log, lvds: types.LVDS, transfer: types.DataTransfer,
        capture: types.DataCapture, format: types.DataFormat
    ) -> None:
        """Configure FPGA."""
        self.log.info("Configuring FPGA: {}, {}, {}, {}, {}".format(
            log, lvds, transfer, capture, format))
        cfg = struct.pack(
            "HHHHHH", log.value, lvds.value, transfer.value,
            capture.value, format.value, types.FPGA_CONFIG_DEFAULT_TIMER)
        cmd = CommandRequest(types.Command.CONFIG_FPGA, cfg)
        resp = self.config_request(cmd)
        if resp.status == types.Status.SUCCESS:
            self.log.info("Configured FPGA.")
        else:
            self.log.error("FPGA config returned failure status.")

    # untested
    def config_eeprom(self) -> None:
        pass

    # untested
    def start_record(self) -> None:
        pass

    # untested
    def stop_record(self) -> None:
        pass

    # untested
    def stop_record_async(self) -> None:
        pass

    def read_fpga_version(self) -> None:
        """Get current FPGA version."""
        cmd = CommandRequest(types.Command.READ_FPGA_VERSION, bytes())
        resp = self.config_request(cmd)
        if resp.status < 0:
            self.log.error(
                "Unable to read FPGA version: {}".format(resp.status))
        else:
            major = resp.status & 0x7F
            minor = 0x7f & (resp.status >> 7)
            playback = resp.status & 0x4000
            self.log.info(
                "FPGA Version: {}.{} [mode={}]".format(
                    major, minor, "playback" if playback else "record"))

    # untested
    def config_data_packet(self) -> None:
        pass

    def config_request(self, cmd: CommandRequest) -> CommandResponse:
        """Send config command."""
        payload = cmd.to_bytes()
        self.config_socket.sendto(payload, self.config_addr)
        self.log.debug("Sent: {}".format(cmd))

        raw, _ = self.config_socket.recvfrom(self._MAX_PACKET_SIZE)
        response = CommandResponse.from_bytes(raw)
        self.log.debug("Received: {}".format(response))
        return response
            

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
dca = DCA1000EVM()
dca.system_aliveness()
dca.read_fpga_version()
dca.config_fpga(
    log=types.Log.RAW_MODE, lvds=types.LVDS.FOUR_LANE,
    transfer=types.DataTransfer.CAPTURE, format=types.DataFormat.BIT16,
    capture=types.DataCapture.ETH_STREAM)
