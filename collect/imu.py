"""IMU data collection."""

import os
from imu_api import XsensIMU, IMUData
from common import BaseCapture, BaseSensor, SensorMetadata


class IMUCapture(BaseCapture):
    """IMU capture data."""

    def _init(self, path: str, **_) -> SensorMetadata:
        _meta = {
            "rot": {
                "format": "raw", "type": "f32", "shape": (3,),
                "desc": "Orientation (Euler Angles)"},
            "acc": {
                "format": "raw", "type": "f32", "shape": (3,),
                "desc": "Linear acceleration"},
            "avel": {
                "format": "raw", "type": "f32", "shape": (3,),
                "desc": "Angular velocity"}
        }
        self.outputs = {
            k: open(os.path.join(path, k), mode='wb') for k in _meta}
        return _meta

    def write(self, data: IMUData) -> None:
        """Write IMU data."""
        for k, v in self.outputs.items():
            v.write(getattr(data, k))

    def close(self) -> None:
        """Close files and clean up."""
        for v in self.outputs.values():
            v.close()
        super().close()


class IMU(BaseSensor):
    """Xsens IMU Sensor.
    
    Parameters
    ----------
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
            except Exception as e:
                self.log.critical(repr(e))

        out.close()

    def close(self):
        pass


if __name__ == '__main__':
    IMU.main()
