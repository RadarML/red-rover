"""Full radar capture system API."""

from .awr_api import AWR1843
from .dca_api import DCA1000EVM


class AWRSystem:
    """Radar capture system with a AWR1843Boost and DCA1000EVM."""

    def __init__(
        self, port: str = "/dev/ttyACM0", sys_ip: str = "192.168.33.30",
        fpga_ip: str = "192.168.33.180", packet_delay: float = 25.0,
        frame_length: int = 64, adc_samples: int = 256, fps: float = 10.0
    ) -> None:

        self.awr = AWR1843(port=port)
        self.dca = DCA1000EVM(sys_ip=sys_ip, fpga_ip=fpga_ip)
        self.dca.setup(delay=packet_delay)

        self.frame_length = frame_length
        self.adc_samples = adc_samples
        self.fps = fps

    def stream(self):
        """Get frame iterator."""
        self.dca.start()
        self.dca.reset_ar_device()
        self.awr.setup(
            frame_length=self.frame_length,
            adc_samples=self.adc_samples, frame_period=1000.0 / self.fps)
        self.awr.start()

        return self.dca.stream([self.frame_length, 3, 4, self.adc_samples, 2])

    def stop(self):
        """Stop data collection."""
        self.awr.stop()
        self.dca.stop()
