# Red Rover Data Processing System

## Quick Start

Assuming you have `python3.11`, `python3.11-venv`, and `python3-pip` installed:
```sh
make env
```

## Setup

1. Install Python 3.11 (`ouster-sdk` does not support `python>=3.12` at present):
    ```sh
    conda create -n rover python=3.11
    conda install pip
    # or
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.11
    sudo apt install python3.11-venv
    ```

2. Install JAX manually to include GPU support:
    ```sh
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -r requirements.txt
    ```

3. (Optional) Set up aliases:
    ```sh
    source rover.sh
    roverp <command> <args...>
    ```
