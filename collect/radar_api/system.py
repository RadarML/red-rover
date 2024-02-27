"""Full radar capture system API."""

import logging

from .awr_api import AWR1843
from .dca_api import DCA1000EVM
from .config import RadarConfig, CaptureConfig


class AWRSystem:
    """Radar capture system with a AWR1843Boost and DCA1000EVM."""

    def __init__(
        self, *, radar: RadarConfig, capture: CaptureConfig,
        name: str = "RadarCapture"
    ) -> None:
        self.log = logging.getLogger(name)
        self._statistics(radar, capture)

        self.dca = DCA1000EVM(
            sys_ip=capture.sys_ip, fpga_ip=capture.fpga_ip,
            data_port=capture.data_port, config_port=capture.config_port,
            timeout=capture.timeout, socket_buffer=capture.socket_buffer)
        self.dca.setup(delay=capture.delay)
        self.awr = AWR1843(port=radar.port)

        self.config = radar
        self.fps = 1000.0 / radar.frame_period * radar.frame_length

    def _statistics(self, radar: RadarConfig, capture: CaptureConfig) -> None:
        """Compute statistics, and warn if potentially invalid."""

        # Network utilization
        util = radar.throughput / capture.throughput
        self.log.info("Radar/Capture card: {} Mbps / {} Mbps ({:.1f}%)".format(
            int(radar.throughput / 1e6), int(capture.throughput / 1e6),
            util * 100))
        if radar.throughput > capture.throughput * 0.8:
            self.log.warn(
                "Network utilization > 80%: {:.1f}%".format(util * 100))

        # Buffer size
        ratio = capture.socket_buffer / radar.frame_size
        self.log.info("Recv buffer size: {:.2f} frames".format(ratio))
        if ratio < 2.0:
            self.log.warn("Recv buffer < 2 frames: {} (1 frame = {})".format(
                capture.socket_buffer, radar.frame_size))

        # Radar duty cycle
        duty_cycle = radar.frame_time / radar.frame_period
        self.log.info("Radar duty cycle: {:.1f}%".format(duty_cycle * 100))
        if duty_cycle > 0.95:
            self.log.warn(
                "Radar duty cycle > 95%: {:.1f}%".format(duty_cycle * 100))

        # Ramp timing
        excess = (
            radar.ramp_end_time - radar.adc_start_time - radar.sample_time)
        self.log.info("Excess ramp time: {:.1f}us".format(excess))
        if excess < 0:
            self.log.warn("Excess ramp time < 0: {:.1f}us".format(excess))

    def stream(self):
        """Get frame iterator."""
        # Reboot radar in case it is stuck
        self.dca.reset_ar_device()
        self.dca.start()
        self.awr.setup(**self.config.as_dict())
        self.awr.start()

        return self.dca.stream(self.config.shape[1:])

    def stop(self):
        """Stop data collection."""
        self.dca.stop()

        # The radar might be ignoring responses if the frame timings are
        # too tight. Just reboot the radar instead.
        self.dca.reset_ar_device()
