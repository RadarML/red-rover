"""Minimal implementation of the Xsens API.

References
----------
[1] Xsens MT Low Level Documentation
    https://www.xsens.com/hubfs/Downloads/Manuals/MT_Low-Level_Documentation.pdf
"""

import logging
import serial
import struct
import numpy as np

from beartype.typing import NamedTuple, Optional
from jaxtyping import Float32


class InvalidChecksum(Exception):
    """Non-fatal IMU error message."""
    pass


class IMUData(NamedTuple):
    """IMU orientation, angular velocity, acceleration data."""

    rot: Float32[np.ndarray, "3"]
    acc: Float32[np.ndarray, "3"]
    avel: Float32[np.ndarray, "3"]


class XsensIMU:
    """Xsens MTi-3 IMU API.
    
    The IMU should be set up using MT Manager with the following:
    - Orientation: Euler Angles; Floating Point 32-bit; 100Hz
    - Inertial data: Rate of Turn, Acceleration; Floating Point 32-bit; 100Hz

    Parameters
    ----------
    port: serial port to use. Must have read/write permissions, e.g.
        `sudo chmod 666 /dev/ttyUSB0`.
    baudrate: serial baudrate.
    name: human readable name.
    """

    PRE = 0xfa
    BID = 0xff
    MTData2 = 0x36

    def __init__(
        self, port: str = "/dev/ttyUSB0", baudrate: int = 115200,
        name: str = "Xsens.MTi-3"
    ) -> None:
        self.log = logging.getLogger(name=name)
        self.port = serial.Serial(port, baudrate, timeout=1)
        self.port.set_low_latency_mode(True)
        self.port.reset_input_buffer()
    
    def read(self) -> Optional[IMUData]:
        """Read a single data packet.

        NOTE: Only handles MTData2 messages [1, p. 45].

        Returns
        -------
        IMUData on success; otherwise, returns None.
        """
        # Fast forward until we get to the preamble start bytes `fa ff``
        prev, current = None, None
        while (prev, current) != (self.PRE, self.BID):
            prev = current
            current = struct.unpack('>B', self.port.read(1))[0]

        mid, packet_len = struct.unpack('>BB', self.port.read(2))
        payload = self.port.read(packet_len)
        checksum = self.port.read(1)

        summation = self.BID + mid + packet_len + sum(payload + checksum)
        if (summation & 0xff) != 0:
            raise InvalidChecksum(summation & 0xff)

        # We only handle MTData2 messages.
        if mid != self.MTData2:
            return None

        rot, acc, avel = None, None, None
        while len(payload) > 0:
            data_id, data_len = struct.unpack('>HB', payload[:3])
            payload = payload[3:]

            data = payload[:data_len]
            payload = payload[data_len:]

            if data_id & 0xfff0 == 0x2030:
                rot = np.frombuffer(data, dtype='>f4').byteswap()
            elif data_id & 0xfff0 == 0x4020:
                acc = np.frombuffer(data, dtype='>f4').byteswap()
            elif data_id & 0xfff0 == 0x8020:
                avel = np.frombuffer(data, dtype='>f4').byteswap()

        return IMUData(rot=rot, acc=acc, avel=avel)  # type: ignore
