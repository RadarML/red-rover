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
        
    '''
    -1 1 1 0 0 0 1
    A B C D E F G

    ???? idk where this is
    
    '''

# cfarCfg
    '''


    mmw_cli.c line 1356

    cliCfg.tableEntry[cnt].cmd            = "cfarCfg";
    cliCfg.tableEntry[cnt].helpString     = "<subFrameIdx> <procDirection> <averageMode> <winLen> <guardLen> <noiseDiv> <cyclicMode> <thresholdScale> <peakGroupingEn>";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLICfarCfg;

    

    mmw_cli.c line 467
    cfarCfg -1 0 2 8 4 3 0 15 1
    cfarCfg -1 1 0 4 2 3 1 15 1
    A B C D E F G H I

    A is proc direction (either 0 for Range or -1 Doppler)
    B is 
    C is window length



    /* Initialize configuration: */
    memset ((void *)&cfarCfg, 0, sizeof(cfarCfg));

    /* Populate configuration: */
    procDirection             = (uint32_t) atoi (argv[2]);
    cfarCfg.averageMode       = (uint8_t) atoi (argv[3]);
    cfarCfg.winLen            = (uint8_t) atoi (argv[4]);
    cfarCfg.guardLen          = (uint8_t) atoi (argv[5]);
    cfarCfg.noiseDivShift     = (uint8_t) atoi (argv[6]);
    cfarCfg.cyclicMode        = (uint8_t) atoi (argv[7]);
    threshold                 = (float) atof (argv[8]);
    cfarCfg.peakGroupingEn    = (uint8_t) atoi (argv[9]);

    if (threshold > 100.0)
    {
        CLI_write("Error: Maximum value for CFAR thresholdScale is 100.0 dB.\n");
        return -1;
    }  

    
    '''

    #can we just tweak this directly?



# multiObjBeamForming
    
    '''
    A B C


    
    '''

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
    
    '''
    A B C

    "<save enable> <restore enable> <Flash offset>";
    #control/mmwavelink/test/common/link_test.c line 3418
    
    '''


# sensorStop
    '''
    call at end, no args

    cliCfg.tableEntry[cnt].cmd            = "sensorStop";
    cliCfg.tableEntry[cnt].helpString     = "No arguments";
    cliCfg.tableEntry[cnt].cmdHandlerFxn  = MmwDemo_CLISensorStop;
    
    '''
        

# frameCfg
        
    '''
    cli_mmwave.c line 826

    this takes the form 
    0 2 16 0 200 1 0
    A B C  D E   F G where each letter is an integer

    E is from the frame rate

    A is chirp start index as int 
    B is chirp end index as int
    C is number of loops
    D is number of frames
    E is frame periodicity
    F is trigger select 
    G is frame trigger delay 

    defaults!
    control/mmwave/test/xwr18xx/full/common_full.c line 631

    ptrCtrlCfg->u.frameCfg.frameCfg.chirpStartIdx      = 0;
    ptrCtrlCfg->u.frameCfg.frameCfg.chirpEndIdx        = 0;
    ptrCtrlCfg->u.frameCfg.frameCfg.numLoops           = 128;
    ptrCtrlCfg->u.frameCfg.frameCfg.numFrames          = 1;
    ptrCtrlCfg->u.frameCfg.frameCfg.numAdcSamples      = 256;
    ptrCtrlCfg->u.frameCfg.frameCfg.framePeriodicity   = 20 * 1000000 / 5;
    ptrCtrlCfg->u.frameCfg.frameCfg.triggerSelect      = 1;
    ptrCtrlCfg->u.frameCfg.frameCfg.frameTriggerDelay  = 0;
    
    '''

# cfarFovCfg

    '''
    A B C D

    C is negative maximum radial velocity
    D is positive maximum radial velocity



    I have no idea where this is ingested 
    all info so far comes from the JS on the demo website -T
    '''


# chirpCfg
    
    '''
    0 0 0 0 0 0 0 1
    A B C D E F G H

    #from utils/cli/src/cli_mmwave.c  

    # A is chirp start index
    # B is chirp end index

    # A and B should be the same????

    # C is profile ID? 

    # I'm 90% sure it should be 0, 
    # it keeps being indexed when constructing profileCfg

    # D is start frequency in Hz (should probably be 0)
    # E is frequency slope in KHz/us (should probably be 0)

    # F is Idle Time in us (default is 0)
    # G is ADC Start Time in us (default 0)

    # H is TX Enable in binary 
    # (1/2/4) are bitmasked to TX 0/1/2
    # chirpCfg should be called three times, with each arg
    
    '''








'''

% Frequency:77
% Platform:xWR18xx_AOP
% Scene Classifier:best_range_res
% Azimuth Resolution(deg):30 + 38
% Range Resolution(m):0.044
% Maximum unambiguous Range(m):9.02
% Maximum Radial Velocity(m/s):1
% Radial velocity resolution(m/s):0.13
% Frame Duration(msec):100
% RF calibration data:None
% Range Detection Threshold (dB):15
% Doppler Detection Threshold (dB):15
% Range Peak Grouping:enabled
% Doppler Peak Grouping:enabled
% Static clutter removal:disabled
% Angle of Arrival FoV: Full FoV
% Range FoV: Full FoV
% Doppler FoV: Full FoV
% ***************************************************************
sensorStop
flushCfg
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 77 267 7 57.14 0 0 70 1 256 5209 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 16 0 100 1 0
lowPower 0 0
guiMonitor -1 1 1 0 0 0 1
cfarCfg -1 0 2 8 4 3 0 15 1
cfarCfg -1 1 0 4 2 3 1 15 1
multiObjBeamForming -1 1 0.5
clutterRemoval -1 0
calibDcRangeSig -1 0 -5 8 256
extendedMaxVelocity -1 0
lvdsStreamCfg -1 0 0 0
compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
measureRangeBiasAndRxChanPhase 0 1.5 0.2
CQRxSatMonitor 0 3 5 121 0
CQSigImgMonitor 0 127 4
analogMonitor 0 0
aoaFovCfg -1 -90 90 -90 90
cfarFovCfg -1 0 0 8.92
cfarFovCfg -1 1 -1 1.00
calibData 0 0 0
sensorStart


'''