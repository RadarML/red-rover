"""Lidar data collection."""

import os
import subprocess
import lzma
from multiprocessing.pool import ThreadPool
from beartype.typing import Optional

import numpy as np
from ouster.sdk import client

from .common import BaseCapture, BaseSensor, SensorException, SensorMetadata


class LidarCapture(BaseCapture):
    """Lidar capture data."""

    def _init(
        self, path: str, shape: tuple[int, int] = (64, 2048),
        compression: int = 0, **_
    ) -> SensorMetadata:
        _meta = {
            "rfl": {
                "format": "lzma", "type": "u1", "shape": shape,
                "desc": "Object NIR reflectivity"},
            "nir": {
                "format": "lzma", "type": "u2", "shape": shape,
                "desc": "Near infrared ambient photons"},
            "rng": {
                "format": "lzma", "type": "u2", "shape": shape,
                "desc": "Range, in millimeters"}}

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
    
    Args:
        addr: lidar IP address; found automatically via
            `avahi-browse -lrt _roger._tcp` if not manually specified.
        port_lidar: lidar port; default 7502.
        port_imu: integrated imu port; default 7503.
        mode: lidar mode `{columns}x{fps}`.
        beams: number of lidar beams.
        name: sensor name, i.e. "lidar".
    """

    def __init__(
        self, addr: Optional[str] = None, port_lidar: int = 7502,
        port_imu: int = 7503, mode: str = "2048x10", beams: int = 64,
        name: str = "lidar"
    ) -> None:
        super().__init__(name=name)

        addr = self.get_ip() if addr is None else addr
        if addr is None:
            self.log.critical("No Ouster lidar found; is it connected?")
            raise SensorException
        self.addr = addr

        config = client.SensorConfig()
        config.udp_port_lidar = port_lidar
        config.udp_port_imu = port_imu
        config.operating_mode = client.OperatingMode.OPERATING_NORMAL
        config.lidar_mode = client.LidarMode.from_string(mode)

        self.fps = float(config.lidar_mode.frequency)
        self.shape = (beams, config.lidar_mode.cols)

        client.set_config(self.addr, config, persist=True, udp_dest_auto=True)
        self.log.info("Initialized lidar {}: {}-beam x {} x {} fps".format(
            self.addr, *self.shape, self.fps))

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
            fps=self.fps, shape=self.shape, compression=0)

        stream = client.Scans.stream(
            hostname=self.addr, lidar_port=7502, complete=True, timeout=1.0)
        with open(os.path.join(path, self.name, "lidar.json"), 'w') as f:
            f.write(stream.metadata.updated_metadata_string())

        for scan in stream:
            out.start()
            data = {
                "rfl": scan.field(
                    client.ChanField.REFLECTIVITY).astype(np.uint8),
                "nir": scan.field(client.ChanField.NEAR_IR).astype(np.uint16),
                "rng": np.minimum(
                    65535, scan.field(client.ChanField.RANGE)
                ).astype(np.uint16)
            }
            out.write(data)  # type: ignore
            out.end()

            if not self.active:
                break

        out.close()
