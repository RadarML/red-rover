from ouster import client
import os
import numpy as np
import subprocess
import lzma
from multiprocessing.pool import ThreadPool
from beartype.typing import Optional

from common import BaseCapture


class LidarCapture(BaseCapture):
    """Lidar capture data.
    
    Parameters
    ----------
    path: directory path to save to.
    fps: target framerate.
    compression: compression level; may need to be lowered until utilization
        is reliably <1. Set `compression=1` for the i7-1360p.
    """

    _META = {
        "rfl": {
            "format": "lzma", "type": "u8", "shape": (64, 2048),
            "description": "Object NIR reflectivity"},
        "nir": {
            "format": "lzma", "type": "u16", "shape": (64, 2048),
            "description": "Near infrared ambient photons"},
        "rng": {
            "format": "lzma", "type": "u16", "shape": (64, 2048),
            "description": "Range, in millimeters"}
    }

    def __init__(
        self, path: str, fps: float = 10.0, compression: int = 1
    ) -> None:
        super().__init__(path, meta=self._META, fps=fps)

        self.outputs = {
            k: lzma.open(
                os.path.join(path, k), mode='wb', preset=compression
            ) for k in self._META}

    def write(self, data: dict[str, np.ndarray]) -> None:
        """Write compressed lzma streams."""
        def _compress(args):
            k, v = args
            self.outputs[k].write(v.tobytes())
        ThreadPool(3).map(_compress, list(data.items()))

    def close(self) -> None:
        """Close files and clean up."""
        super().close()
        for v in self.outputs.values():
            v.close()


class Lidar:

    def __init__(
        self, addr: Optional[str] = None, port_lidar: int = 7502,
        port_imu: int = 7503, fps: float = 10.0
    ) -> None:
        self.addr = self.get_ip() if addr is None else addr
        self.fps = fps

        config = client.SensorConfig()
        config.udp_port_lidar = port_lidar
        config.udp_port_imu = port_imu
        config.operating_mode = client.OperatingMode.OPERATING_NORMAL
        client.set_config(self.addr, config, persist=True, udp_dest_auto=True)

    @staticmethod
    def get_ip() -> str:
        """Get IP address of ouster sensor."""
        avahi = subprocess.run(
            ["avahi-browse", "-lrpt", "_roger._tcp"], capture_output=True)
        for service in avahi.stdout.decode('utf-8').split('\n'):
            entries = service.split(';')
            match = (
                entries[0] == '=' and
                entries[2] == 'IPv4' and
                "Ouster" in entries[3] and
                len(entries) > 8)
            if match:
                return entries[7]
        else:
            raise Exception("Ouster Lidar not found.")

    def capture(self, path: str) -> None:
        out = LidarCapture(path, fps=self.fps, compression=1)

        stream = client.Scans.stream(
            hostname=self.addr, lidar_port=7502, complete=True, timeout=1.0)
        def _read(scan, field):
            return client.destagger(stream.metadata, scan.field(field))

        for i, scan in enumerate(stream):
            out.start()
            data = {
                "rfl": client.destagger(
                    stream.metadata, scan.field(client.ChanField.REFLECTIVITY)
                ).astype(np.uint8),
                "nir": client.destagger(
                    stream.metadata, scan.field(client.ChanField.NEAR_IR)
                ).astype(np.uint16),
                "rng": np.minimum(65535, client.destagger(
                    stream.metadata, scan.field(client.ChanField.RANGE))
                ).astype(np.uint16)
            }
            out.write(data)
            out.end()

            if (i + 1) % 20 == 0:
                out.reset_stats()

            if i > 10 * 30:
                break

        out.close()

    def close(self):
        pass


# Lidar().capture("test/lidar")
