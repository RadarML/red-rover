from ouster import client
import os
import numpy as np
import subprocess
import time
import lzma
from multiprocessing.pool import ThreadPool
from beartype.typing import Optional

from stats import DutyCycle


class Lidar:

    def __init__(
        self, addr: Optional[str] = None, port_lidar: int = 7502,
        port_imu: int = 7503, fps: float = 10
    ) -> None:
        self.addr = self.get_ip() if addr is None else addr
        self.fps = fps

        print(self.addr)

        config = client.SensorConfig()
        config.udp_port_lidar = port_lidar
        config.udp_port_imu = port_imu
        config.operating_mode = client.OperatingMode.OPERATING_NORMAL
        client.set_config(self.addr, config, persist=True, udp_dest_auto=True)

        self.meta = {
            "rfl": {
                "format": "lzma", "type": "u8", "shape": (64, 2048),
                "description": "Object NIR reflectivity"},
            "nir": {
                "format": "lzma", "type": "u16", "shape": (64, 2048),
                "description": "Near infrared ambient photons"},
            "rng": {
                "format": "lzma", "type": "u16", "shape": (64, 2048),
                "description": "Range, in millimeters"},
            "ts": {
                "format": "raw", "type": "f64", "shape": (),
                "description": "System epoch time"}
        }

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

        os.makedirs(path)

        out = {
            k: lzma.open(os.path.join(path, k), mode='wb')
            for k in ["rfl", "nir", "rng"]
        }

        stream = client.Scans.stream(
            hostname=self.addr, lidar_port=7502, complete=True, timeout=1.0)

        def _read(scan, field):
            return client.destagger(stream.metadata, scan.field(field))

        stats = DutyCycle()

        for i, scan in enumerate(stream):
            t = time.time()
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

            def _compress(args):
                k, v = args
                out[k].write(lzma.compress(v.tobytes()))

            ThreadPool(3).map(_compress, list(data.items()))            

            stats.observe((time.time() - t) * self.fps)
            if i > 100:
                break

        stats.print()

        for k, v in out.items():
            v.close()


Lidar().capture("test_lidar")
