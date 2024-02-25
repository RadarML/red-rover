import logging
from radar_api import AWR1843

logging.basicConfig(level=logging.DEBUG)

awr = AWR1843()

awr.stop()
awr.flushCfg()
awr.dfeDataOutputMode()
awr.channelCfg()
awr.adcCfg()
awr.adcbufCfg()
awr.profileCfg()
awr.chirpCfg()
awr.chirpCfg(chirpStartIdx=1,chirpEndIdx=1,txEnable=2)
awr.chirpCfg(chirpStartIdx=2,chirpEndIdx=2,txEnable=4)
awr.frameCfg()
awr.lowPower()
awr.guiMonitor()
awr.cfarCfg(procDirection=0,averageMode=2,winLen=8,guardLen=4,cyclicMode=0)
awr.cfarCfg()
awr.multiObjBeamForming()
awr.clutterRemoval()
awr.calibDcRangeSig()
awr.extendedMaxVelocity()
awr.lvdsStreamCfg()
awr.compRangeBiasAndRxChanPhase()
awr.measureRangeBiasAndRxChanPhase()
awr.CQRxSatMonitor()
awr.CQSigImgMonitor()
awr.analogMonitor()
awr.aoaFovCfg()
awr.cfarFovCfg(subFrameIdx=-1, procDirection=0, min_meters_or_mps=0, max_meters_or_mps=8.92)
awr.cfarFovCfg(subFrameIdx=-1, procDirection=1, min_meters_or_mps=-1, max_meters_or_mps=1.00)
awr.calibData()
awr.start()


print("done!")






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