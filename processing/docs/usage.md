# Usage

## Common Recipes

Get dataset metadata:
```sh
roverp info -p path/to/dataset
```

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
```

Prepare DART data:
```sh
export DATASET=dataset
roverp rosbag -p data/$DATASET
make lidar
roverp sensorpose -p data/$DATASET -s radar
roverp fft -p data/$DATASET --mode hybrid
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

## Scripts

Run `roverp <command> -h` for more information, or see [the full documentation](scripts.rst).

- `align`: Get sensor timestamp alignments as indices.
   - any set of sensors &rarr; `_fusion/indices.npz` unless overridden.
- `as_nerfstudio`: Convert poses and video into Nerfstudio format.
   - `calibration/camera_ns.json`, `camera/video.avi`, `_camera/pose.npz` &rarr; `_nerfstudio`
- `as_rover1`: Convert to legacy DART format.
   - `radar/`, `_radar/pose.npz`, `_radar/rover1` &rarr; `_rover1/*`
- `cfar`: Run CFAR and AOA estimation.
   - `radar/*` &rarr; `_cfar/*`
- `cfar_map`: Create reflectance grid from cfar data.
   - `_cfar/points.npz`, `_slam/trajectory.csv` &rarr; `_cfar/map.npz`.
- `cfar_pointcloud`: Create CFAR point cloud.
   - `_cfar/*`, `_radar/pose.npz` &rarr; `_cfar/pointcloud.npz`.
- `compare`: Render simulation (novel view synthesis) comparison video.
   - any set of radar-like data in `_radar`, or elsewhere so long as they match the format of `_radar/rda`. &rarr; `_report/compare.mp4` unless overridden.
- `decompress`: Export decompressed lidar data.
   - `lidar/*` &rarr; `_lidar/*`, unless overridden.
- `export`: Export dataset to another location.
   - An entire dataset. &rarr; The specified `--out` directory.
- `fft`: Run range-doppler-azimuth FFT.
   - `radar/*`, `_radar/pose.npz` (optional) &rarr; `_radar/{mode}` depending on the selected mode.
- `info`: Get dataset metadata.
   - `/*` &rarr; Printed to `stdout`.
- `lidarmap`: Create ground truth occupancy grid.
   - `_slam/trajectory.csv`, `_slam/lidar.bag_points.ply` &rarr; `_rover1/map.npz` or `_slam/map.npz`, depending on `--legacy`.
- `nearest`: Run nearest-neighbor simulation.
   - `_radar/pose.npz`, `_radar/rda` &rarr; `_radar/sim_nearest` (same shape, properties as the reference channel).
- `report`: Get speed report.
   - `_radar/pose.npz`, `_slam/trajectory.csv` &rarr; `_report/speed.pdf`
- `rosbag`: Convert Rover dataset to ROS 1 bag for Cartographer.
   - `lidar/*` &rarr; `_scratch/lidar.bag`
- `sensorpose`: Get interpolated poses for a specific sensor.
   - `_slam/trajectory.csv` &rarr; `{sensor}/pose.npz` depending on the specified `--sensor`.
- `simulate`: Run radar simulations using an occupancy or reflectance grid.
   - `_radar/pose.npz`, `_slam/map.npz` &rarr; `_radar/sim_lidar` or `_radar/sim_cfar` (same shape, properties as the reference channel).
- `slice`: Render map slices.
   - any map-like data, e.g. `_slam/map.npz`. &rarr; `_report/slices.mp4` unless overridden.
- `video`: Render sensor data video.
   - `camera/*`, `lidar/*`, `_radar/{radar}` for the specified `--radar`. &rarr; `_report/data.mp4`

Generate this table with `scripts/_summary_table.py`.
