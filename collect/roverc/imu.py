"""IMU data collection."""

import logging
import os
from queue import Queue

from .common import Capture, Sensor
from .imu_api import IMUData, InvalidChecksum, XsensIMU


class IMUCapture(Capture):
    """IMU capture data."""

    def __init__(
        self, path: str, fps: float = 1.0,
        report_interval: float = 5.0, log: logging.Logger | None = None
    ) -> None:
        super().__init__(
            path=path, fps=fps, report_interval=report_interval, log=log)

        self.outputs: dict[str, Queue] = {
            channel: Queue() for channel in ["rot", "acc", "avel"]}
        self.sensor.create("rot", {
            "format": "raw", "type": "f4", "shape": (3,),
            "desc": "Orientation (Euler Angles)"
        }).consume(self.outputs["rot"], thread=True)
        self.sensor.create("acc", {
            "format": "raw", "type": "f4", "shape": (3,),
            "desc": "Linear acceleration"
        }).consume(self.outputs["acc"], thread=True)
        self.sensor.create("avel", {
            "format": "raw", "type": "f4", "shape": (3,),
            "desc": "Angular velocity"
        }).consume(self.outputs["avel"], thread=True)

    def queue_length(self) -> int:
        return max(v.qsize() for v in self.outputs.values())

    def write(self, data: IMUData) -> None:
        """Write IMU data."""
        for k, v in self.outputs.items():
            v.put(getattr(data, k))

    def close(self) -> None:
        """Close files and clean up."""
        for v in self.outputs.values():
            v.put(None)
        super().close()


class IMU(Sensor):
    """Xsens IMU Sensor.

    Args:
        port: serial port to use. Must have read/write permissions, e.g.
            `sudo chmod 666 /dev/ttyUSB0`.
        baudrate: serial baudrate.
        fps: imu reading rate (Hz).
        name: sensor name, i.e. "imu".
    """

    def __init__(
        self, port: str = "/dev/ttyUSB0", baudrate: int = 115200,
        fps: float = 100.0, name: str = "imu"
    ) -> None:
        super().__init__(name=name)

        self.fps = fps
        self.imu = XsensIMU(port=port, baudrate=baudrate)
        self.log.info("Initialized IMU {}.".format(port))

    def capture(self, path: str) -> None:
        """Create capture (while `active` is set).

        NOTE: we discard the first 100 samples due to rate instability, likely
        due to serial port read latency/batching.
        """
        out = IMUCapture(
            os.path.join(path, self.name), log=self.log, fps=self.fps)
        self.imu.port.reset_input_buffer()
        while self.active:
            try:
                data = None
                while data is None:
                    data = self.imu.read()
                out.start()
                out.write(data)
                out.end()
            except InvalidChecksum:
                self.log.warn("Received invalid checksum.")
            except Exception as e:
                self.log.critical(repr(e))

        out.close()

    def close(self):
        pass
