
# Setup Instructions

## Radar (Windows) Computer

0. If using RDP to control the data collection system, set up a RDP connection on the Windows laptop.
    1. Open the `Remote Desktop Connection` program (should be a default program on Windows).
    2. Enter the computer name that you've set, i.e. `dart-radar.local`, and your username.
    3. (Optional) Create a shortcut: click `Show Options`, then `Save As...`.

1. If manual time synchronization is required, set up the Windows computer as a NTP server.
    - You can also use the linux computer as a NTP server or a common external NTP server (preferred). Only follow these instructions if that is not possible.
    - See full instructions [here](https://techlibrary.hpe.com/docs/otlink-wo/How-to-Configure-a-Local-NTP-Server.html).

2. Install mmWave studio & dependencies. See the [DCA1000EVM Quick Start Guide](https://www.ti.com/tool/DCA1000EVM) for full instructions.
    - Make sure to set a static IP address (`192.168.30.33`) for the network interface used as shown in step 3.
    - Make sure to install the matlab runtime engine noted in step 4.
    - You may need to install the [mmWave SDK](https://www.ti.com/tool/MMWAVE-SDK).
    - Make sure that all 6 COM ports are detected as shown in Step 6 / Figure 4. If the `XDS110 Class` ports are not detected, see the note about installing [EMUPACK](http://processors.wiki.ti.com/index.php/XDS_Emulation_Software_Package).
    - *CMU Internal*: ask Tianshu for a pre-downloaded collection of all required installation files.
    - **NOTE**: you may need to disable windows firewall, which might block the ports used by the radar by default.
    - **NOTE**: keep a copy of the mmWave installer. If the radar computer loses power while mmWave studio is running, it may become corrupted in a subtle way and require reinstallation (the symptoms we observed were a "null reference" error when trying to start data collection (but not initialize the radar/capture card), and mmWave studio failing to read the FPGA version).
    - **NOTE**: both the DCA1000EVM and the AWR1843Boost need to be fully plugged in during the installation process; otherwise, mmWave Studio will refuse to load full until started with both connected.

3. Clone this repository and install dependencies.
    ```sh
    git clone git@github.com:thetianshuhuang/rover
    ```
    - **NOTE**: submodules (i.e. recursive cloning) are not used by the radar data collection system.
    - Install Python >3.8, and `pip install -r radar/requirements.txt`.

4. Configure the lua scripts.
    - Note which COM port is labeled "XDS110 Class Application/User UART". You can verify that you have selected the correct one using mmWave studio:
        1. Open mmWave studio.
        2. Click the `Set` button under `Board Control / Reset Control / Reset`. Wait for the board to finish resetting.
        3. Select the COM port in the `Board Control / RS232 Operations` dropdown.
        4. Click `Connect`. The `RS232 Connectivity Status` field should turn to `Connected`.
    - Copy sample configuration `rover/radar/_config.json` to `rover/radar/config.json`, and update it with your configuration parameters.
        - Make sure to set the `com` entry with the COM port number noted above.
        - Unless you have a non-standard mmWave Studio installation, none of the other fields need to be changed.
    - Build the lua scripts:
        ```sh
        python build.py
        ```

## LIDAR (Ubuntu 20.04) Computer

0. If using SSH to control the data collection system, make sure to install/enable SSH.
    - A virtual terminal client like `screen`, `tmux`, `zellij` is also helpful. We use `screen` in these instructions (`sudo apt-get -y install screen`).

1. Synchronize time if required (`sudo apt-get install -y ntpdate` if needed):
    ```sh
    sudo ntpdate dart-radar.local
    ```

2. Set `Link-Local` only. In the GUI:
    1. Go to Settings/Network. Click on the ''settings'' button next to the wired connection.
    2. In the IPv4 tab, set `IPv4 Method` to `Link-Local Only`.
    3. In the IPv6 tab, set `IPv6 Method` to `Link-Local Only`.
    4. Disable and re-enable the wired connection.

3. Install [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
    - **NOTE**: make sure you activate the ROS environment on each terminal you use to run subsequent ROS commands:
        ```sh
        source /opt/ros/noetic/setup.bash
        ```

4. Clone this repository. Make sure to clone all subrepositories!
    ```sh
    git clone git@github.com:thetianshuhuang/rover --recursive
    ```
    -  Use `git submodule update --init --recursive` if you forgot to `git clone --recursive`.

5. Install ROS dependencies.
    - Install dependencies for [ouster-ros](https://github.com/ouster-lidar/ouster-ros).
    - Install `python3-catkin-tools`:
        ```sh
        sudo apt-get -y install python3-catkin-tools
        ```
    - Follow manual installation instructions in `lidar/catkin_ws/src/xsens_ros_mti_driver`:
        ```sh
        pushd src/xsens_ros_mti_driver/lib/xspublic && make && popd
        ```
6. Build:
    ```sh
    source /opt/ros/noetic/setup.bash
    catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release
    ```
    - Use `catkin clean` to delete all built files if a rebuild is needed.
    - `--cmake-args -DCMAKE_BUILD_TYPE=Release` is needed for `ouster-ros`.

## Data Processing Computer

Install the following on a fast computer, ideally with both good single-core CPU performance and a GPU.

**Cartographer**: You're on your own for this one.

0. [ROS Noetic](http://wiki.ros.org/noetic) is required.
1. [Cartographer ROS](https://github.com/cartographer-project/cartographer_ros) placed inside the currently sourced `catkin_ws`.

**Python**: The python environment for [DART](https://github.com/thetianshuhuang/dart) is a superset of `rover`, so you can use that if you have one set up.

0. Ensure that you have python (>=3.8) and CUDA (>=11.8).
1. Install [jax](https://github.com/google/jax).
2. Install `libhdf5`: ```sudo apt-get -y install libhdf5-dev```
3. Install python dependencies: ```pip install -r processing/requirements.txt```
