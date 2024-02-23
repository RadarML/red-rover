# Setup Guide

## Radar

### System overview

Red Rover is based on the TI AWR1843Boost mmWave radar and the DCA1000EVM capture card. The AWR1843Boost
has a LVDS (Low Voltage Differential Signaling) "debug" port which outputs data being sent from the radar back-end to an onboard DSP processor, which can be sent to the capture card; the capture card contains a FPGA, which buffers this data and translates it into ethernet packets.

The AWR1843Boost device firmware has three code sections:
1. MSS (Master Sub-System): high-level control of the radar, which runs on an onboard ARM Cortex R4F. We use the demo firmware provided with mmWave SDK, which can be found at `demo/xwr18xx/mmwave/xwr18xx_mmw_demo.bin`.
2. DSS (DSP Sub-System): control code for the onboard DSP. This section is combined with the MSS code in the compiled code distributed with mmWave SDK.
3. RSS/BSS (Radar Sub-System / Backend Sub-System): low-level control of the radar. Uses TI proprietary code, which can be found in `firmware/radarss/xwr18xx_radarss_rprc.bin` in a mmWave SDK installation. Note that this is (probably) also included with the MSS code. 

### Flashing the AWR1843Boost

Flash the radar using [TI UniFlash](https://www.ti.com/tool/UNIFLASH); note that it seems to work most reliably on Windows.

1. Set the radar to flash mode.
    - Find `SOP0:2` (DIP switches on the front of the radar).
    - Set the switches to `SOP0:2=101`, where 1 corresponds to the "on" position labeled on the PCB.
2. Flash using UniFlash.
    - Uniflash should automatically discover the radar.
    - Select the `demo/xwr18xx/mmwave/xwr18xx_mmw_demo.bin` image to flash.
    - Choose the serial port corresponding to the radar; the serial port should have a name/description "XDS110 Class Application/User UART (COM3)".
    - Flashing should take around 1 minute, and terminate with "Program Load completed successfully".
    - If the SOP switches are not in the correct position, flashing will fail with
        > Not able to connect to serial port. Recheck COM port selected and/or permissions
3. Set the radar to functional mode.
    - Set `SOP0:2=001`.
    - Note that mmWave studio expects the radar to be in *debug* mode (`SOP0:2=011`), so switching between Red Rover and mmWave Studio requires the position of the SOP switches to be changed. This is also why mmWave studio requires the MSS firmware to be "re-flashed" whenever the radar is rebooted.

### Hardware Troubleshooting

The [TI mmWave Demo Visualizer](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/) is a good way to validate hardware functionality.
- The demo visualizer uses the same firmware that Red Rover uses.
- If an error occurs, it may be due to a hardware fault. If a specific error occurs, it should be raised with a line number in `mss_main.c`; the error case (e.g. `RL_RF_AE_CPUFAULT_SB`) should reveal what general type of fault it is.


## Camera

Plug a HDMI cable into a monitor in order to change the settings. Select the following using the buttons on the front:
- Focus: set to "optimum focus" - move the focus lever into the detent at the bottom.
- Gain: 18db (change based on lighting conditions).
- Output: clean output, 60Hz.

After closing the menu, the HDMI output should be "clean," and not show any menu items.

## IMU

Install [MT Manager](https://www.movella.com/support/software-documentation), and connect the Xsens MTi-3 IMU via the development board. Select the following:
- Orientation: Euler Angles; Floating Point 32-bit; 100Hz
- Inertial data: Rate of Turn, Acceleration; Floating Point 32-bit; 100Hz
