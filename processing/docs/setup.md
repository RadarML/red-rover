# Setup

## Quick Start

Assuming you have `python3.12`, `python3.12-venv`, and `python3-pip` installed:
```sh
make env
```

Set up alias:
```sh
source rover.sh
```

## Full Setup

1. Install Python 3.12:
    ```sh
    conda create -n rover python=3.12
    conda activate rover
    conda install pip
    # or
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.12
    sudo apt install python3.12-venv
    ```

2. Install libraries.

    If using `venv`:
    ```sh
    make env
    ```

    If using `conda`:
    ```sh
    pip install -r requirements.txt
    ```

3. (Optional) Set up aliases (requires `venv`):
    ```sh
    source rover.sh
    ```
    This will also assign `export $ROVERP=...` to allow calling `roverp` in scripts, nq, and makefiles as `$(ROVERP)`:
    ```sh
    roverp <command> <args...>
    nq $ROVERP <command> <args...>
    ```

4. (Optional) Set up [Cartographer/Ros/Docker stack](docker.md).

    After setting up docker, you can build the catkin workspace:
    ```sh
    make docker
    # once in the docker container...
    cd rover/catkin_ws
    catkin_make
    ```
