# Software Setup and Sensor Configuration

## Computer

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

1. Install [uv](https://github.com/astral-sh/uv):

    ```sh
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # or
    wget -qO- https://astral.sh/uv/install.sh | sh
    ```

2. Install `red-rover`:
    ```sh
    git clone git@github.com:RadarML/red-rover.git
    cd red-rover/collect
    uv sync
    ```

## Sensors

**Radar**: Follow the setup instructions described in the [XWR documentation](https://radarml.github.io/xwr/setup/#awr1843boost).

**Camera**: Plug a HDMI cable into a monitor in order to change the settings. Select the following using the buttons on the front:

- Focus: set to "optimum focus" - move the focus lever into the detent at the bottom.
- Gain: 18db (for typical indoors lighting; change based on lighting conditions).
- SDI output: 1080p60

    !!! note

        The HDMI menu overlay is not shown on the SDI output.

**IMU**: Install [MT Manager](https://www.movella.com/support/software-documentation), and connect the Xsens MTi-3 IMU via the development board. Select the following:

- Orientation: Euler Angles; Floating Point 32-bit; 100Hz
- Inertial data: Rate of Turn, Acceleration; Floating Point 32-bit; 100Hz

**Lidar**: Should work out-of-the-box.

!!! warning

    Make sure that an IP is assigned to the lidar in the [web management page](https://static.ouster.dev/sensor-docs/image_route1/image_route2/connecting/connecting-to-sensors.html#sensor-web-interface).
