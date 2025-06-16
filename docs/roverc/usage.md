# Manual Operation

!!! info

    These steps may be useful for debugging and/or troubleshooting.

## Makefile + Zellij

0. Set configuration file.
    ```sh
    ROVER_CFG=`realpath config/dart.yaml`
    ```

1. Configure interfaces.
    ```sh
    sudo -E make config
    ```

    !!! info

        We must use `-E` in order to pass through environment variables (i.e. `ROVER_CFG`). You can also use `sudo ROVER_CFG=path/to/cfg.yaml make config`.

2. Start data collection processes.
    ```sh
    make
    ```
    
    !!! warning
    
        This requires/uses [zellij](https://zellij.dev/):
        ```sh
        sudo snap install zellij --classic
        ```

3. Run data collection
    ```sh
    make start
    make stop
    ```

4. Clean up and kill data collection processes by quitting zellij (`ctrl` + `q`).

## Fully Manual

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
    screen -dm bash -c "./env/bin/python cli.py run --config [config.yaml] --sensor radar"
    screen -dm bash -c "./env/bin/python cli.py run --config [config.yaml] --sensor camera"
    screen -dm bash -c "./env/bin/python cli.py run --config [config.yaml] --sensor lidar"
    screen -dm bash -c "./env/bin/python cli.py run --config [config.yaml] --sensor imu"
    ```

3. Run data collection
    ```
    ./env/bin/python cli.py start -c [config.yaml] -p [output/path]
    ./env/bin/python cli.py stop
    ```

4. Terminate each data collection process
    ```
    ./env/bin/python cli.py exit -c [config.yaml]
    ```
