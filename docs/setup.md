# Setup Guide

## Computer

The data collection computer uses a linux installation; we use ubuntu 22.04, though other versions should work as well.

TODO: try this procedure and write up the steps.
- Python (deadsnakes 3.11)
    ```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.11
    sudo apt install python3.11-venv
    ```
- Pip
- Helpful: htop, zellij, openssh-server, git

## Radar

A new radar needs to be configured with the following:
1. Flash firmware to the Radar (AWR1843Boost - red board), and set the DIP switch to "functional" mode.
2. Configure DIP switches on the Capture Card (DCA1000EVM - green board).

### System overview

*Red Rover* is based on the TI AWR1843Boost mmWave radar and the DCA1000EVM capture card. The AWR1843Boost
has a LVDS (Low Voltage Differential Signaling) "debug" port which outputs data being sent from the radar back-end to an onboard DSP processor, which can be sent to the capture card; the capture card contains a FPGA, which buffers this data and translates it into ethernet packets.

The AWR1843Boost device firmware has three code sections:
1. MSS (Master Sub-System): high-level control of the radar, which runs on an onboard ARM Cortex R4F. We use the demo firmware provided with mmWave SDK, which can be found at `demo/xwr18xx/mmwave/xwr18xx_mmw_demo.bin`.
2. DSS (DSP Sub-System): control code for the onboard DSP. This section is combined with the MSS code in the compiled code distributed with mmWave SDK.
3. RSS/BSS (Radar Sub-System / Backend Sub-System): low-level control of the radar. Uses TI proprietary code, which can be found in `firmware/radarss/xwr18xx_radarss_rprc.bin` in a mmWave SDK installation. Note that this is (probably) also included with the MSS code. 

### Flashing the AWR1843Boost

Flash the radar using [TI UniFlash](https://www.ti.com/tool/UNIFLASH); note that it seems to work most reliably on Windows. Also, obtain a copy of the `xwr18xx_mmw_demo.bin` firmware (e.g. through installing [mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK)).

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

### Configure Capture Card

Ensure that the following DIP switches are set:
- SW2.5: `SW_CONFIG`
- SW2.6: `USER_SW1` (the marked right side), unless the EEPROM is messed up from
    a misconfigured `configure_eeprom(...)` call.

The following are configured by the `configure_fpga(...)` call under normal operation, but can be manually set in case that isn't working:
- SW1: 16-bit mode (`16BIT_ON`, `14BIT_OFF`, `12BIT_OFF`).
- SW2.1: `LVDS_CAPTURE`
- SW2.2: `ETH_STREAM`
- SW2.3: `AR1642_MODE` (2-lane LVDS)
- SW2.4: `RAW_MODE`
- SW2.5: `HW_CONFIG`

### Hardware Troubleshooting

The [TI mmWave Demo Visualizer](https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/) is a good way to validate hardware functionality, and uses the same firmware that Red Rover uses.

Possible faults:
- An error is returned on the console in the Demo Visualizer: there may be a hardware fault. It should be raised with a line number in `mss_main.c`; the error case (e.g. `RL_RF_AE_CPUFAULT_SB`) should reveal what general type of fault it is.
- When powered on, the capture card error lights should all come on for ~1sec, then turn off again. If this does not occur, the FPGA may be dead.

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

## Lidar

Should work out-of-the-box.
