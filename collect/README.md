# Red Rover Data Collection System

See the [setup instructions](/docs/setup.md).

## UI Operation

1. Specify configuration file and start server:
    ```sh
    ROVER_CFG=`dart.yaml` make server
    ```

2. Select the output path.
    - Dropdown: output location (locally in the `./data` folder or in an external drive)
    - Text input: output dataset name. If blank, will be named after the current timestamp in `YYYY-MM-DD.HH-MM-SS` format.

3. Use the start/stop buttons.
    - The **Start** button will be disabled after starting data collection until the **Stop** button is pressed. This status may not be accurate if there are multiple connections or the page is refreshed.
    - The **Stop** button will be greyed out if the page thinks no data collection sessions are active, but the `stop` command will still be sent even if clicked in case a desync has occured since there is no harm in attempting to stop when there is no active session.

## CLI Operation

0. Set configuration file.
    ```sh
    ROVER_CFG=`realpath dart.yaml`
    ```

1. Configure interfaces.
    ```sh
    sudo -E make config
    ```
    - **NOTE**: we must use `-E` in order to pass through environment variables (i.e. `ROVER_CFG`). You can also use `sudo ROVER_CFG=path/to/cfg.yaml make config`.

2. Start data collection processes.
    ```sh
    make
    ```
    - **NOTE**: this requires/uses [zellij](https://zellij.dev/).

3. Run data collection
    ```sh
    make start
    make stop
    ```

4. Clean up and kill data collection processes by quitting zellij (`ctrl` + `q`).


## Manual Operation

1. Configure interfaces:
    - Set the radar interface to a static IP:
        ```sh
        sudo ifconfig [radar/interface] 192.168.33.30 netmask 255.255.255.0
        ```
    - Set the socket receive buffer size:
        ```sh
        echo [radar/args/capture/socket_buffer] | sudo tee /proc/sys/net/core/rmem_max
        ```
    - Set Lidar interface to link-local:
        ```
        sudo ip addr add 169.254.11.103/16 dev [lidar/interface]
        ```
    - Set IMU, radar serial port permissions:
        ```sh
        sudo chmod 666 /dev/ttyACM0
        sudo chmod 666 /dev/ttyUSB0
        ```

2. Launch data collection processes (in the background, in separate terminals, in `screen`, etc.):
    ```sh
    screen -dm bash -c "./env/bin/python collect.py run -c [config.yaml] -s radar"
    screen -dm bash -c "./env/bin/python collect.py run -c [config.yaml] -s camera"
    screen -dm bash -c "./env/bin/python collect.py run -c [config.yaml] -s lidar"
    screen -dm bash -c "./env/bin/python collect.py run -c [config.yaml] -s imu"
    ```

3. Run data collection
    ```
    ./env/bin/python collect.py start -c [config.yaml] -p [output/path]
    ./env/bin/python collect.py stop
    ```

4. Terminate each data collection process
    ```
    ./env/bin/python collect.py exit -c [config.yaml]
    ```
