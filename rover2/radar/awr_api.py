"""AWR1843 TI Demo API [3]."""

from enum import Enum
import serial
import logging


class LVDSFormat(Enum):
    """LVDS data format.
     
    See `mmw_config.h:MmwDemo_LvdsStreamCfg` [3].
    """

    DISABLED = 0
    ADC = 1
    _RESERVED2 = 2
    _RESERVED3 = 3
    CP_ADC_CQ = 4


class AWR1843:
    """AWR1843 Interface for the TI `demo/xwr18xx` MSS firmware.
 
    Documented by [3]; based on a UART ASCII CLI.

    Usage
    -----
    """

    def __init__(
        self, port: str = "/dev/ttyACM0", name: str = "AWR1843",
        baudrate: int = 115200
    ) -> None:
        self.log = logging.getLogger(name=name)
        self.port = serial.Serial(port, baudrate, timeout=1)

    def setup(self) -> None:
        self.config_lvds()

    def send(self, cmd: str) -> None:
        self.log.info("Send: {}".format(cmd))
        self.port.write((cmd + '\n').encode('ascii'))

    def start(self, reconfigure: bool = True) -> None:
        """Start radar.
        
        Parameters
        ----------
        reconfigure: Whether the radar needs to be configured.
        """
        if reconfigure:
            self.send("sensorStart")
        else:
            self.send("sensorStart 0")

    def stop(self) -> None:
        """Stop radar."""
        self.send("sensorStop")

# guiMonitor

# cfarCfg

# multiObjBeamForming

# calibDcRangeSig

# clutterRemoval

# adcbufCfg

# compRangeBiasAndRxChanPhase

# measureRangeBiasAndRxChanPhase

# aoaFovCfg

# cfarFovCfg

# extendedMaxVelocity

# CQRxSatMonitor

# CQSigImgMonitor

# analogMonitor

    def config_lvds(
        self, subframe: int = -1, enable_header: bool = True,
        data_format: LVDSFormat = LVDSFormat.ADC, sw_enabled: bool = False
    ) -> None:
        """Configure LVDS stream (to the DCA1000EVM); `LvdsStreamCfg`.

        Parameters
        ----------
        subframe: subframe to apply to. If `-1`, applies to all subframes.
        enable_header: HSI (High speed interface; refers to LVDS) Header
            enabled/disabled flag; only applies to HW streaming. Must be
            enabled for the DCA1000EVM [4].
        data_format: LVDS format; we assume `LVDSFormat.ADC`.
        sw_enabled: Use software (SW) instead of hardware streaming; causes
            chirps to be streamed during the inter-frame time after processing.
            We assume HW streaming.

        References
        ----------
        [4] TI forums: https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/845372/dca1000evm-how-to-relate-data-sent-from-dca1000-through-ethernet-to-the-data-sent-from-awr1843-through-uart
        """
        cmd = "lvdsStreamCfg {} {} {} {}".format(
            subframe, 1 if enable_header else 0, data_format.value,
            1 if sw_enabled else 0)
        self.send(cmd)

# configDataPort

# queryDemoStatus

# calibData
