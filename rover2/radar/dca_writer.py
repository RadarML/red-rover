"""Data writer."""

import os
import logging
import struct
import time
import json
from io import TextIOWrapper

from .dca_types import DataPacket


class RadarDataWriter:
    """Radar data writer.
    
    NOTE: `RadarDataWriter` is agnostic to the antenna layout; `chirp` refers
    to a single TX-RX pair. The data writer chirp rate should be the overall
    per-pair chirp rate multiplied by the number of pairs.

    Parameters
    ----------
    path: destination path (folder to contain raw files)
    name: writer name (used for logging)
    chirp_len: chirp len, in uint16 I/Q samples (used for chirp timekeeping)
    buffer: file buffer size, in chirps
    """

    def __init__(
        self, path: str, name: str = "RadarDataWriter", chirp_len: int = 512,
        buffer: int = 4096
    ) -> None:
        self.log = logging.getLogger(name=name)
        self.chirp_len = chirp_len
        self.path = path

        self.offset = -1
        self.radar_iq = self._open("radar_iq", buffer * chirp_len)
        self.radar_ts = self._open("radar_ts", buffer)

        datatypes = {
            "radar_iq": {"type": "uint16", "shape": [chirp_len]},
            "radar_ts": {"type": "float64", "shape": []}}
        with open(os.path.join(path, "metadata.json"), 'w') as f:
            json.dump(datatypes, f, indent=4)


    def _open(self, path: str, buf: int) -> TextIOWrapper:
        path = os.path.join(self.path, path)
        if os.path.exists(path):
            raise ValueError("File already exists: {}".format(path))
        return open(path, 'a', buf)

    def _write_time(self) -> None:
        self.radar_ts.write(struct.pack("d", time.time()))

    def write(self, packet: DataPacket) -> None:
        """Write data to file.

        Notes
        -----
        - `self.offset` is set to the next expected byte index; any gaps are
          filled with `0xff` (i.e. 0xffff after interpreting as u16).
        - `radar_time` is written with a timestamp of when the first
          corresponding packet is received.
        """
        if self.offset < 0:
            initial = packet.byte_count - (packet.byte_count % self.chirp_len)
            if initial > 0:
                self._write_time()        
            self.offset = initial

        # Fill in blank packets
        if packet.byte_count > self.offset:
            size = packet.byte_count - self.offset
            self.log.warn("Missing I/Q data: {} bytes".format(size))
            self.radar_iq.write(b'\xff' * size)
 
        # Write data
        self.radar_iq.write(packet.data)
        offset_new = self.offset + len(packet.data)

        # Write timestamps
        chirp_start = self.offset + (-self.offset % self.chirp_len)
        for _ in range(chirp_start, offset_new, self.chirp_len):
            self._write_time()

        # Commit
        self.offset = offset_new

    def close(self):
        """Safely clean up."""
        self.radar_data.close()
        self.radar_time.close()
