# Setup Guide

## Hardware

![Wiring diagram](wiring.svg)

!!! info

    Listed prices are as of August 2024, and may be out of date.

??? quote "3D Printed Parts"

    | Item | Qty | Description | 
    | ---- | --- | ----------- |
    | `top-plate.stl` | 1 | Mount for Lidar, IMU |
    | `front-plate.stl` | 1 | Primary vertical structure |
    | `mid-plate.stl` | 1 | Connects top plate to the NUC VESA mounting plate |
    | `bottom-plate.stl` | 1 | Provides structural support, and a tray to put cables in |
    | `handle-bracket.stl` | 2 | Joins vertical and horizontal handles |
    | `radar-cover.stl` | 1 | Protective cover for the radar; should be covered in ESD tape / other material |

??? quote "Core Components"

    | Cost | Qty | Description |
    | ---- | --- | ----------- |
    | $299.00 | 1 | [TI AWR1843Boost](https://www.ti.com/tool/AWR1843BOOST) Radar + 5V3A barrel jack power supply if not included |
    | $599.00 | 1 | [TI DCA1000EVM](https://www.ti.com/tool/DCA1000EVM) Capture Card |
    | ~$1,000 | 1 | NUC, with included VESA mounting plate |
    | ~$8,000-$12,000 | 1 | [Ouster OS0-64 or OS0-128](https://ouster.com/products/hardware/os0-lidar-sensor) |
    | $449.00 | 1 | [Xsens MTi-3](https://shop.movella.com/us/product-lines/sensor-modules/products/mti-3-ahrs-development-kit) |
    | $995.00 | 1 | [Black Magic Micro Studio Camera (BMMSC)](https://www.bhphotovideo.com/c/product/1787638-REG/blackmagic_design_micro_studio_camera_4k.html) |
    | $299.00 | 1 | [Magewell 1080p60 USB-SDI capture card](https://www.bhphotovideo.com/c/product/1350328-REG/magewell_32070_usb_3_0_sdi_capture.html) |
    | $99.0 | 1 | [Olympus Fisheye Body Cap Lens](https://www.bhphotovideo.com/c/product/1026132-REG/olympus_v325040bw000_bcl_0980_fisheye_body_cap.html)
    | $28.00 | 1 | [Micro BNC to BNC cable](https://www.amazon.com/HangTon-Female-Adapter-Blackmagic-Monitor/dp/B09BJQNDNP) |
    | $6.29 | 1 | [BNC Coupler](https://www.amazon.com/TLS-eagle-Coaxial-Coupler-Straight-Connector/dp/B083LZ39HM/) |
    | $31.90 | 1 | [Router](https://www.amazon.com/GL-iNet-GL-AR300M16-Ext-Pre-Installed-Performance-Programmable/dp/B07794JRC5) + power supply, power cable if not included |
    | $9.99 | 1 | [USB A Hub](https://www.bhphotovideo.com/c/product/1496562-REG/xcellon_usb_4311_2_slim_4_port_usb_3_1.html) |
    | - | 1 | AC battery bank; should be able to supply >200W. |
    | - | 2 | 6" Cat 5/6 Ethernet cable |
    | - | 2 | 6" micro USB to USB A type 2.0 cable
    | - | 1 | Power strip |


??? quote "Mechanical Parts"

    | Cost | Qty | Description |
    | ---- | --- | ----------- |
    | $19.90 | 2 | [Top handles](https://www.bhphotovideo.com/c/product/1736879-REG/smallrig_1638c_top_handle_1_4_20_screws.html) |
    | $29.90 | 2 | [Side handles](https://www.bhphotovideo.com/c/product/1689008-REG/smallrig_3813_mini_nato_side_handle.html) |
    | $9.90 | 2 | [Mounting Rail](https://www.bhphotovideo.com/c/product/1502679-REG/smallrig_1195b_quick_release_safety_rail.html) |
    | - | - | 1/4-20" socket cap screws, nuts |
    | - | - | M3 standoffs, socket cap screws |
    | - | - | M4 socket cap screws, nuts |

??? quote "Optional Accessories"

    | Cost | Qty | Description |
    | ---- | --- | ----------- |
    | $329.99 | 2 | [External SSD](https://www.bhphotovideo.com/c/product/1595436-REG/sandisk_sdssde81_4t00_g25_4tb_extremepro_portable_ssd.html) |
    | $349 | 1 | ["Lapdock" (screen + keyboard + trackpad)](https://www.amazon.com/NexDock-Touchscreen-Wireless-Portable-Compatible/dp/B0CSK2T47Q/) |

## Software

### Computer

The data collection computer uses a linux installation; we use ubuntu 22.04, though other versions should work as well.

0. Install linux.

    !!! info

        If using a usb C monitor / dock to debug (e.g. the [NexDock](https://nexdock.com/explore-nexdock/)), you may need to install [DisplayLink drivers](https://www.synaptics.com/products/displaylink-graphics/downloads/ubuntu) in order to allow hot-plugging the dock.

    - Internet access is required during setup.
    - Other helpful packages:
        ```sh
        sudo apt install -y htop openssh-server git
        sudo snap install zellij --classic
        ```
    - Make sure that all ethernet interfaces are enabled (3 in total), and that the lidar interface is set to "link-local only".

1. Install dependencies:
    ```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install -y python3.12 python3.12-venv build-essential net-tools
    ```

2. Install `red-rover`:
    ```sh
    git clone git@github.com:WiseLabCMU/red-rover.git
    cd red-rover/collect
    make env
    ```

### Sensors

**Radar**: Follow the setup instructions described in the [XWR documentation](https://wiselabcmu.github.io/xwr/setup/#awr1843boost).

**Camera**: Plug a HDMI cable into a monitor in order to change the settings. Select the following using the buttons on the front:

- Focus: set to "optimum focus" - move the focus lever into the detent at the bottom.
- Gain: 18db (for typical indoors lighting; change based on lighting conditions).
- SDI output: 1080p60

Note that the HDMI menu overlay is not shown on the SDI output.

**IMU**: Install [MT Manager](https://www.movella.com/support/software-documentation), and connect the Xsens MTi-3 IMU via the development board. Select the following:

- Orientation: Euler Angles; Floating Point 32-bit; 100Hz
- Inertial data: Rate of Turn, Acceleration; Floating Point 32-bit; 100Hz

**Lidar**: Should work out-of-the-box. Make sure that an IP is assigned to the lidar in the management page.
