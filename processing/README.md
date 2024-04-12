# Red Rover Data Processing System

![Data processing pipeline](/docs/processing.svg)

## Quick Start

Assuming you have `python3.11`, `python3.11-venv`, and `python3-pip` installed:
```sh
make env
```


## Setup

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
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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

4. (Optional) Set up [Cartographer/Ros/Docker stack](/docs/docker.md).


## Common Recipes

Upload data to storage server or external drive:
```sh
roverp export -p {path/to/dataset} -o {path/to/destination}
```

Prepare radarhd data:
```sh
export SRC=path/to/dataset
export DST=path/to/destination

roverp export -p $SRC -o $DST --metadata
roverp align -p $DST --mode left
roverp decompress -p $SRC -o $DST
cp $SRC/radar/iq $DST/radar/iq
cp -r $SRC/_fusion $DST/_fusion
```

Prepare DART data:
```sh
export DATASET=path/to/dataset
roverp rosbag -p $DATASET
make lidar
roverp sensorpose -p $DATASET -s radar
roverp fft -p $DATASET --mode hybrid
```

Generate reports (requires DART data):
```sh
export DATASET=path/to/dataset
roverp report -p $DATASET
roverp video -p $DATASET
```

Generate simulations (requires DART data):
```sh
export DATASET=path/to/dataset
roverp nearest -p $DATASET
roverp lidarmap -p $DATASET
roverp simulate -p $DATASET
```

## Commands

Run `roverp <command> -h` for more information.

| Command | Description | Inputs | Outputs |
| ------- | ----------- | ------ | ------- |
| `as_rover1` | Convert to legacy DART format. | `radar/`, `_radar/pose.npz`, `_radar/rover1`  | `_rover1/*`  |
| `align` | Get sensor timestamp alignments as indices. | any set of sensors  | `_fusion/indices.npz` unless overridden.  |
| `compare` | Create simulation (novel view synthesis) comparison video. | any set of radar-like data in `_radar`, or elsewhere so long as they match the format of `_radar/rda`.  | `_report/compare.mp4` unless overridden.  |
| `export` | Export dataset to another location. | an entire dataset.  | the dataset is copied into the `dst` directory (e.g. on an external drive or file server), except for easily recreated processed output files.  |
| `fft` | Generate range-doppler-azimuth images. | `_radar/pose.npz`, `radar/*`  | `_radar/{mode}` depending on the selected mode.  |
| `info` | Print dataset metadata. | `/*`  | Printed to `stdout`.  |
| `lidarmap` | Create ground truth occupancy grid. | `_slam/trajectory.csv`, `_slam/lidar.bag_points.ply`  | `_rover1/map.npz` or `_slam/map.npz`, depending on `--legacy`.  |
| `nearest` | Run nearest-neighbor simulation. | `_radar/pose.npz`, `_radar/rda`  | `_radar/sim_nearest`  |
| `report` | Get speed report. | `_radar/pose.npz`, `_slam/trajectory.csv`  | `_report/speed.pdf`  |
| `rosbag` | Convert Rover dataset to ROS 1 bag for Cartographer. | `lidar/*`  | `_scratch/lidar.bag`  |
| `sensorpose` | Calculate interpolated poses for a specific sensor. | `_slam/trajectory.csv`  | `{sensor}/pose.npz` depending on the specified `--sensor`.  |
| `simulate` | Simulate radar range-doppler data. | `_radar/pose.npz`, `_slam/map.npz`  | `_radar/sim_lidar`  |
| `slice` | Render map slices. | any map-like data, e.g. `_slam/map.npz`.  | `_report/slices.mp4` unless overridden.  |
| `video` | Create sensor data video. | `camera/*`, `lidar/*`, `_radar/rda`  | `_report/data.mp4`  |

Generate this table with `scripts/_summary_table.py`.
