"""Full radar capture system API."""

from .awr_api import AWR1843
from .dca_api import DCA1000EVM


class AWRSystem:
    """Radar capture system with a AWR1843Boost and DCA1000EVM."""

    def __init__(
        self, port: str = "/dev/ttyACM0", sys_ip: str = "192.168.33.30",
        fpga_ip: str = "192.168.33.180", packet_delay: float = 5.0,
        config: dict = {}
    ) -> None:

        self.dca = DCA1000EVM(sys_ip=sys_ip, fpga_ip=fpga_ip)
        self.dca.setup(delay=packet_delay)
        self.awr = AWR1843(port=port)

        self.config = config
        self.fps = config.get('fps', 10.0)
        self.shape = [
            config.get('frame_length', 16), 3, 4,
            config.get('adc_samples', 256), 2]

    def stream(self):
        """Get frame iterator."""
        # Reboot radar in case it is stuck
        self.dca.reset_ar_device()
        self.dca.start()
        # self.awr.setup(
        #     frequency=77.0, idle_time=107.9,
        #     adc_start_time=5.99, ramp_end_time=58.22,
        #     tx_start_time=2.0, freq_slope=63.005, 
        #     adc_samples=256, sample_rate=5000,
        #     frame_length=64, frame_period=34.0)
        self.awr.setup(
            frequency=77.0, idle_time=106.0,
            adc_start_time=5.99, ramp_end_time=58.22,
            tx_start_time=2.0, freq_slope=63.005, 
            adc_samples=256, sample_rate=5000,
            frame_length=16, frame_period=8.0)

        self.awr.start()

        return self.dca.stream(self.shape)

    def stop(self):
        """Stop data collection."""
        self.dca.stop()

        # The radar might be ignoring responses if the frame timings are
        # too tight. Just reboot the radar instead.
        self.dca.reset_ar_device()
