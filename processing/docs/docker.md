# Cartographer - Ros - Docker

## Install

1. Ensure docker is installed:
    ```bash
    sudo snap install docker
    sudo apt-get install docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
    ```

2. Install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)  

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-docker2  
    sudo systemctl restart docker  
    ```

3. Test the install (may require a reboot, especially if `nvidia-smi` returns an error):
    ```sh
    sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
    ```

## Running

These steps are taken care of by `make docker-run` in `collect`. Follow these steps for manual troubleshooting:

1. Ensure that X11 forwarding is correctly set (if this does not work, you may need to activate non-local X11 forwarding).
    ```
    xhost +local:docker
    ```

2. Run the container with the requisite NVIDIA driver arguments, the desktop env variable, and the link to the unix X11 "file". 
    ```bash
    sudo docker run -it --privileged --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix qoschatz/cartographer_ros /bin/bash
    ```

    Mount any desired folders by adding
    ```
    -v /{full_path_to_folder}:/{full_path_in_container}
    ```

    **NOTE**: Pipe all processing results to a mounted directory to prevent outputs from being deleted when the docker container stops running.

## Troubleshooting
A common error is a failure from the `nvidia-container-cli` noting that it was unable to load `libnvidia-ml` because of `no such file or directory`. This is caused by NVIDIA hating you, and the best fix seems to be to reinstall `docker.io` (i.e. the docker engine). If that does not work, reinstall everything else as well, which will typically resolve this issue.

*(Tianshu, 06/2024)* - I suspect this may be caused by some kind of time issue affecting certificates, triggering automated invalidations/reinstalls/etc. This behavior was seen in our demo PC, which had CMOS battery issues. The immediate symptom was total failure of the Docker/Nvidia stack; additional indications include the system time being off anddocker images being re-pulled each time instead of cached.
