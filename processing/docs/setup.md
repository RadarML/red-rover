# Setup

## Quick Start

Assuming you have `python3.11`, `python3.11-venv`, and `python3-pip` installed:
```sh
make env
```

Set up alias:
```sh
source rover.sh
```

## Full Setup

1. Install Python 3.11 (`ouster-sdk` does not support `python>=3.12` at present):
    ```sh
    conda create -n rover python=3.11
    conda activate rover
    conda install pip
    # or
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.11
    sudo apt install python3.11-venv
    ```

2. Install libraries.

    If using `venv`:
    ```sh
    make env
    ```

    If using `conda`:
    ```sh
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install -r requirements.txt
    ```

    **NOTE**: if not using a GPU, set `JAX_CUDA=cpu`, e.g.
    ```sh
    JAX_CUDA=cpu make env
    # or
    pip install --upgrade "jax[cpu]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
