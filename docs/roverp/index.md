# **roverp**: data processing

The data processing tooling for `red-rover` includes some library components, a number of CLI tools, and a makefile for managing a dockerized cartographer SLAM pipeline.

| Tool | Description | Required Extras |
| ---- | ----------- | ---------------- |
| [`roverp anonymize`](./cli.md#roverp-anonymize) | Anonymize camera data by blurring faces. | `anonymize` |
| [`roverp sensorpose`](./cli.md#roverp-sensorpose) | Get interpolated poses for a specific sensor from a cartographer output. | |
| [`roverp report`](./cli.md#roverp-report) | Generate a "speed report" of a dataset. | |
| [`roverp segment`](./cli.md#roverp-segment) | Run semantic segmentation on collected video data. | `segment` |
| [`make trajectory`](./docker.md) | Compute poses using cartographer. |
| [`make lidar`](./docker.md) | Compute poses and output globally-aligned point clouds. |
| [`make info`](./docker.md) | Alias for `rosbag info`. |
| [`make validate`](./docker.md) | Alias for `cartographer_rosbag_validate`. |

## Setup

=== "uv + local"

    ```sh
    git clone git@github.com:RadarML/red-rover.git
    cd roverp
    uv sync
    ```

=== "pip + github"

    ```sh
    pip install "roverp@git+ssh://git@github.com/RadarML/red-rover.git#subdirectory=processing"
    ```

!!! warning

    `roverp` includes a number of extras, many of which are fairly heavy, and most of which you probably don't need. Avoid installing unnecessary dependencies by only installing the required extras and definitely don't `uv sync --all-extras`!.

!!! info

    If you need to run cartographer SLAM, see the [dockerized cartographer setup instructions](docker.md).
