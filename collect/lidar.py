"""Lidar data collection."""

import os
import subprocess
import lzma
from multiprocessing.pool import ThreadPool
from beartype.typing import Optional

import numpy as np
from ouster import client

from common import BaseCapture, BaseSensor, SensorException, SensorMetadata


class LidarCapture(BaseCapture):
    """Lidar capture data."""

    def _init(
        self, path: str, height: int = 64, compression: int = 1, **_
    ) -> SensorMetadata:
        _meta = {
            "rfl": {
                "format": "lzma", "type": "u8", "shape": (height, 2048),
                "desc": "Object NIR reflectivity"},
            "nir": {
                "format": "lzma", "type": "u16", "shape": (height, 2048),
                "desc": "Near infrared ambient photons"},
            "rng": {
                "format": "lzma", "type": "u16", "shape": (height, 2048),
                "desc": "Range, in millimeters"},
            "time": {
                "format": "lzma", "type": "f64", "shape": (2048,),
                "desc": "Lidar internal timestamp of each column in the scan"}
        }

        self.outputs = {
            k: lzma.open(os.path.join(path, k), mode='wb', preset=compression)
            for k in _meta}

        return _meta

    def write(self, data: dict[str, np.ndarray]) -> None:
        """Write compressed lzma streams."""
        def _compress(args):
            k, v = args
            self.outputs[k].write(v.tobytes())
        ThreadPool(3).map(_compress, list(data.items()))

    def close(self) -> None:
        """Close files and clean up."""
        for v in self.outputs.values():
            v.close()
        super().close()


class Lidar(BaseSensor):
    """Ouster Lidar sensor.
    
    Parameters
    ----------
    addr: lidar IP address; found automatically via
        `avahi-browse -lrt _roger._tcp` if not manually specified.
    port_lidar: lidar port; default 7502.
    port_imu: integrated imu port; default 7503.
    fps: lidar framerate.
    height: number of lidar beams.
    name: sensor name, i.e. "lidar".
    """

    def __init__(
        self, addr: Optional[str] = None, port_lidar: int = 7502,
        port_imu: int = 7503, fps: float = 10.0, height: int = 64,
        name: str = "lidar"
    ) -> None:
        super().__init__(name=name)

        addr = self.get_ip() if addr is None else addr
        if addr is None:
            self.log.critical("No Ouster lidar found; is it connected?")
            raise SensorException
        self.addr = addr

        self.fps = fps
        self.height = height

        config = client.SensorConfig()
        config.udp_port_lidar = port_lidar
        config.udp_port_imu = port_imu
        config.operating_mode = client.OperatingMode.OPERATING_NORMAL
        client.set_config(self.addr, config, persist=True, udp_dest_auto=True)
        self.log.info("Initialized lidar {}: {}-beam @ {} fps".format(
            self.addr, height, self.fps))

    @staticmethod
    def get_ip() -> Optional[str]:
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
            return None

    def capture(self, path: str) -> None:
        """Create capture (while `active` is set)."""
        out = LidarCapture(
            os.path.join(path, self.name), log=self.log,
            fps=self.fps, height=self.height, compression=1)

        stream = client.Scans.stream(
            hostname=self.addr, lidar_port=7502, complete=True, timeout=1.0)
        for scan in stream:
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
                ).astype(np.uint16),
                "time": scan.timestamp.astype(np.float64)
            }
            out.write(data)  # type: ignore
            out.end()

            if not self.active:
                break

        out.close()


if __name__ == '__main__':
    Lidar.main()
