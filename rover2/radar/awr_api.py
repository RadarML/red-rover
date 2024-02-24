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
        while(True):
            line=self.port.readline()
            print("Response:", line)

            if not line.strip():
                break
        

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

    def flushCfg(self) -> None:
        self.send("flushCfg")


    #dfeDataOutputMode
    
    def dfeDataOutputMode(
            self, 
            modeType: int = 1) -> None: #0 means send no bytes on this data pot
        

        cmd = "dfeDataOutputMode {}".format(modeType)
        self.send(cmd)





# guiMonitor
        
    #all parameters are flags
    #1 is enable and 0 is disable

    def guiMonitor(
            self, 
            subFrameIdx: int = -1, 
            detectedObjects: int = 1, 
            logMagRange: int = 1,
            noiseProfile: int = 1,
            rangeAzimuthHeatMap: int = 1,
            rangeDopplerHeatMap: int = 1, 
            statsInfo: int = 1) -> None:
        

        cmd = "guiMonitor {} {} {} {} {} {} {}".format(
            subFrameIdx, detectedObjects, logMagRange, noiseProfile, rangeAzimuthHeatMap,
            rangeDopplerHeatMap, statsInfo)
        self.send(cmd)

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


    def cfarCfg(
            self, 
            subFrameIdx: int = -1, 
            procDirection: int = 1, 
            averageMode: int = 0,
            winLen: int = 4,
            guardLen: int = 2,
            noiseDivShift: int = 3, 
            cyclicMode: int = 1,
            threshold: float = 15,
            peakGroupingEn: int = 1) -> None:
        

        cmd = "cfarCfg {} {} {} {} {} {} {} {} {}".format(
            subFrameIdx, procDirection, averageMode, winLen, guardLen,
            noiseDivShift, cyclicMode, threshold, peakGroupingEn)
        self.send(cmd)



    def channelCfg(
            self, 
            rxChannelEn: int = 15, #bitmasking
            txChannelEn: int = 7, #bitmasking
            cascading: int = 0 #SoC cascading, should be 0 for now
            ) -> None:
        

        cmd = "channelCfg {} {} {}".format(
            rxChannelEn, txChannelEn, cascading)
        self.send(cmd)


    def adcCfg(
            self, 
            numADCBits: int = 2, 
            adcOutputFmt: int = 1) -> None:
        

        cmd = "adcCfg {} {}".format(
            numADCBits, adcOutputFmt)
        self.send(cmd)



# multiObjBeamForming
    
    def multiObjBeamForming(
            self, 
            subFrameIdx: int = -1, 
            enabled: int = 0, 
            threshold: float = 0.5) -> None:
        

        cmd = "multiObjBeamForming {} {} {}".format(
            subFrameIdx, enabled, threshold)
        self.send(cmd)

# calibDcRangeSig
        
    def calibDcRangeSig(
            self, 
            subFrameIdx: int = -1, 
            enabled: int = 0, 
            negativeBinIdx: int = -5,
            positiveBinIdx: int = 8,
            numAvgFrames: int = 256) -> None:
        

        cmd = "calibDcRangeSig {} {} {} {} {}".format(
            subFrameIdx, enabled, negativeBinIdx, positiveBinIdx, numAvgFrames)
        self.send(cmd)    

# clutterRemoval
        
    def clutterRemoval(
            self, 
            subFrameIdx: int = -1, 
            enabled: int = 0) -> None:
        

        cmd = "clutterRemoval {} {}".format(
            subFrameIdx, enabled)
        self.send(cmd)


# adcbufCfg
        
    def adcbufCfg(
            self, 
            subFrameIdx: int = -1, 
            adcOutputFmt: int = 0, 
            SampleSwap: int = 1,
            ChanInterleave: int = 1,
            ChirpThreshold: int = 1) -> None:
        
        cmd = "adcbufCfg {} {} {} {} {}".format(
            subFrameIdx, adcOutputFmt, SampleSwap, 
            ChanInterleave, ChirpThreshold)
        self.send(cmd)

        

# compRangeBiasAndRxChanPhase
        
    #"<rangeBias> <Re00> <Im00> <Re01> <Im01> <Re02> <Im02> <Re03> <Im03> <Re10> <Im10> <Re11> <Im11> <Re12> <Im12> <Re13> <Im13> ";
    #this one is complicated
    
    def compRangeBiasAndRxChanPhase(
            self, 
            rangeBias: float = 0.0, 
            Re00: int = 1, 
            Im00: int = 0,
            Re01: int = 1, 
            Im01: int = 0,
            Re02: int = 1, 
            Im02: int = 0,
            Re03: int = 1, 
            Im03: int = 0,
            Re10: int = 1, 
            Im10: int = 0,
            Re11: int = 1, 
            Im11: int = 0,
            Re12: int = 1, 
            Im12: int = 0,
            Re13: int = 1, 
            Im13: int = 0) -> None:
        
        cmd = "compRangeBiasAndRxChanPhase {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
            rangeBias, Re00, Im00, Re01, Im01, Re02, Im02, Re03, Im03, 
            Re10, Im10, Re11, Im11,  Re12, Im12, Re13, Im13)
        self.send(cmd)
    
        


# measureRangeBiasAndRxChanPhase
        
    def measureRangeBiasAndRxChanPhase(
            self, 
            enabled: int = 0, 
            targetDistance: int = 0, 
            searchWin: int = 0) -> None:
        

        cmd = "measureRangeBiasAndRxChanPhase {} {} {}".format(
            enabled, targetDistance, searchWin)
        self.send(cmd)
    

# aoaFovCfg

    def aoaFovCfg(
            self, 
            subFrameIdx: int = -1, 
            minAzimuthDeg: int = -90, 
            maxAzimuthDeg: int = 90,
            minElevationDeg: int = -90,
            maxElevationDeg: int = 90) -> None:
        
        cmd = "aoaFovCfg {} {} {} {} {}".format(
            subFrameIdx, minAzimuthDeg, maxAzimuthDeg, 
            minElevationDeg, maxElevationDeg)
        self.send(cmd)    
    

# cfarFovCfg
    
    def cfarFovCfg(
            self, 
            subFrameIdx: int = -1, 
            procDirection: int = 0, 
            min_meters_or_mps: float = 0,
            max_meters_or_mps: float = 0) -> None:
        
        cmd = "cfarFovCfg {} {} {} {}".format(
            subFrameIdx, procDirection, min_meters_or_mps, max_meters_or_mps)
        self.send(cmd)    


# extendedMaxVelocity
        
    def extendedMaxVelocity(
            self, 
            subFrameIdx: int = -1, 
            enabled: int = 0) -> None:
        

        cmd = "extendedMaxVelocity {} {}".format(subFrameIdx, enabled)
        self.send(cmd)

# CQRxSatMonitor 

    def CQRxSatMonitor( #this one is actually enabled
            self, 
            profile: int = 0, 
            satMonSel: int = 3, 
            priSliceDuration: int = 5,
            numSlices: int = 128, 
            rxChanMask: int = 0) -> None:
        
        cmd = "CQRxSatMonitor {} {} {} {} {}".format(
            profile, satMonSel, priSliceDuration, numSlices, rxChanMask)
        self.send(cmd)    
        
    
# CQSigImgMonitor
        
    def CQSigImgMonitor( #this one is actually enabled
            self, 
            profile: int = 0, 
            numSlices: int = 128, 
            numSamplePerSlice: int = 1) -> None:
        
        cmd = "CQSigImgMonitor {} {} {}".format(
            profile, numSlices, numSamplePerSlice)
        self.send(cmd) 
    

# analogMonitor
        
    def analogMonitor(
            self, 
            rxSaturation: int = 0,  #0 is disable
            sigImgBand: int = 0) -> None:
        

        cmd = "analogMonitor {} {}".format(rxSaturation, sigImgBand)
        self.send(cmd)


    def lvdsStreamCfg(
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
    #optional command to change the buadrate
    
    def configDataPort(
            self, 
            baudrate: int = 921600, 
            ackPing: int = 0) -> None: #0 means send no bytes on this data pot
        

        cmd = "configDataPort {} {}".format(baudrate, ackPing)
        self.send(cmd)


    def lowPower(
            self, 
            dontCare: int = 0, 
            adcMode: int = 0) -> None: 
        

        cmd = "lowPower {} {}".format(dontCare, adcMode)
        self.send(cmd)


        


# queryDemoStatus
    #optional command to get sensor state
    def queryDemoStatus(self) -> None:
        """Sensor State"""
        self.send("queryDemoStatus")

# calibData
    #mandatory
    
    '''
    A B C

    "<save enable> <restore enable> <Flash offset>";
    #control/mmwavelink/test/common/link_test.c line 3418
    
    '''
    def calibData(
            self, 
            save_enable: int = 0, 
            restore_enable: int = 0, 
            Flash_offset: int = 0) -> None:
        
        cmd = "calibData {} {} {}".format(
            save_enable, restore_enable, Flash_offset)
        self.send(cmd) 


        

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
    ptrCtrlCfg->u.frameCfg.frameCfg.numFrames          = 1; //0 means infinite
    ptrCtrlCfg->u.frameCfg.frameCfg.numAdcSamples      = 256;
    ptrCtrlCfg->u.frameCfg.frameCfg.framePeriodicity   = 20 * 1000000 / 5;
    ptrCtrlCfg->u.frameCfg.frameCfg.triggerSelect      = 1;
    ptrCtrlCfg->u.frameCfg.frameCfg.frameTriggerDelay  = 0;
    
    '''

    def frameCfg(
            self, 
            chirpStartIdx: int = 0, 
            chirpEndIdx: int = 2, 
            numLoops: int = 16,
            numFrames: int = 0,
            framePeriodicity: float = 100,
            triggerSelect: int = 1, 
            frameTriggerDelay: float = 0) -> None:
        

        cmd = "frameCfg {} {} {} {} {} {} {}".format(
            chirpStartIdx, chirpEndIdx, numLoops, numFrames, framePeriodicity,
            triggerSelect, frameTriggerDelay)
        self.send(cmd)


    


# chirpCfg
    
    '''
    #file:///run/user/1000/gvfs/smb-share:server=shiraz.arena.andrew.cmu.edu,share=shiraz/scratch/ti/control/mmwavelink/docs/doxygen/html/structrl_chirp_cfg__t.html
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

    def chirpCfg(
            self, 
            chirpStartIdx: int = 0, 
            chirpEndIdx: int = 0, 
            profileId: int = 0,
            startFreqVar: float = 0,
            freqSlopeVar: float = 0,
            idleTimeVar: float = 0, 
            adcStartTimeVar: float = 0,
            txEnable: int = 1) -> None:
        

        cmd = "chirpCfg {} {} {} {} {} {} {} {}".format(
            chirpStartIdx, chirpEndIdx, profileId, startFreqVar, freqSlopeVar,
            idleTimeVar, adcStartTimeVar, txEnable)
        self.send(cmd)

    #fix this
    def profileCfg(
            self, 
            profileId: int = 0, 
            startFreq: float = 77, 
            idleTime: float = 267,
            adcStartTime: float = 7,
            rampEndTime: float = 57.14,
            txOutPower: int = 0, #demo only supports 0
            txPhaseShifter: int = 0, #demo only supports 0
            freqSlopeConst: float = 70,
            txStartTime: float = 1,
            numAdcSamples: int = 256,
            digOutSampleRate: int = 5209,
            hpfCornerFreq1: int = 0,
            hpfCornerFreq2: int = 0,
            rxGain: int = 30) -> None:
        

        cmd = "profileCfg {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
            profileId, startFreq, idleTime, adcStartTime, rampEndTime,
            txOutPower, txPhaseShifter, freqSlopeConst, txStartTime, numAdcSamples,
            digOutSampleRate, hpfCornerFreq1, hpfCornerFreq2, rxGain)
        self.send(cmd)







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

'''



'''