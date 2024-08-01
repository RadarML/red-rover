# Dataset Format

## Usage

Dataset are created by `roverc`, and can be copied to a faster computer for processing. After processing, any created files should be stored in directories starting with `_`. These files do not need to be distributed, and can be excluded, e.g.
```sh
tar --exclude '*/_*' -cf data.tar data
```

**NOTE**: the underlying data representations should already be compressed; further compressing the dataset (e.g. `-z`) as a whole is unlikely to yield a nontrivial compression ratio.

## Specifications

**Channel**: a channel (e.g. `iq`, `acc`, `rng`) is a single sequential homogenous data stream from a sensor.
- Each channel is stored as a file, and may be stored in a raw binary format, compressed, or any other format (e.g. `raw`, `lzma`, `mjpeg`).
- Channel names may have file extensions (e.g. `.avi`) if the file format can be interpreted by other programs.
- The "shape" of each channel is described by the constant axes of the data, i.e. the overall array for a channel with `N` observations has shape `(N, *shape)`.
- Channel names must not start with an underscore (`_`).
- Channel names cannot be `meta.json` or `ts`, except for the special "timestamp" channel.

**Sensor**: a sensor (e.g. `imu`, `lidar`, `radar`, `camera`) is a collection of channels, where sequential entries in each channel correspond to a single series of timestamps.
- Sensors are represented by directories, containing one file for each channel.
- Sensor names must not start with an underscore (`_`).
- Each sensor must have a `ts` channel with format `raw` and type `f64`, which contains the epoch time, measured in seconds.
- Sensor directories may contain additional non-channel metadata / sensor intrinsics files other than `meta.json`, e.g. `lidar.json`.
- Each sensor directory must have a `meta.json` file, which includes the data type and file format for each sensor channel, for example:
    ```json
    {
        "iq": {
            "format": "raw", "type": "i2", "shape": [64, 3, 4, 512],
            "desc": "Raw I/Q stream."
        },
        "valid": {
            "format": "raw", "type": "u1", "shape": [],
            "desc": "True if this frame is complete (no zero-fill)."
        },
        "ts": {
            "format": "raw", "type": "f8", "shape": [],
            "description": "Timestamp, in seconds."
        }
    }
    ```
- Data types are specified according to [numpy size-in-bytes convention](https://numpy.org/doc/stable/reference/arrays.dtypes.html).

**Dataset**: a directory containing a collection of sensors.
- A copy of the active `config.yaml` is placed in the root dataset folder.
- A `README` or other metadata files can be manually placed in the root dataset folder. These files should not start with `_`, and should not be referenced in the `config.yaml`.
- Data processing output files should be placed in directories starting with an underscore (e.g. `_scratch`, `_out`).
- Processed output files placed in a directory along with a valid `meta.json` may be interpreted as "virtual" sensors, e.g. a processed range-doppler-azimuth radar sensor.
- **NOTE**: processed directories must not contain a `meta.json` file unless they are intended to be interpreted as virtual sensors.
- Example structure:
    ```
    example/
        _scratch/
            ...
        _slam/
            ...
        camera/
            meta.json
            ts
            vdeo.avi
        imu/
            acc
            avel
            meta.json
            rot
            ts
        lidar/
            lidar.json
            meta.json
            nir
            rfl
            rng
            ts
        radar/
            iq
            meta.json
            ts
            valid
        README
        config.yaml
    ```

## Index of Channels & Files

`config.yaml`: configuration used when the dataset was collected; see the [data collection documentation](../collect/index.html).

`camera`: collected video data.
- `meta.json`: channel metadata.
- `video.avi`: MJPEG-encoded video data. Expected to be 1920x1080 at 30 or 60 fps.
- `ts`: timestamps.

`_camera`: processed video data.
- `meta.json`: channel metadata.
- `clip`: CLIP embeddings for each frame, cut into 8 patches (4 wide x 2 high); see `roverp clip`.
- `ts`: copy of the `camera` timestamps.
- `pose.npz`: poses from the SLAM pipeline aligned to camera frame timestamps; see `roverp sensorpose -s camera`.

`_cfar`: CFAR outputs; see `roverp cfar`.
- `meta.json`: channel metadata.
- `amplitude`: point amplitudes.
- `cfar`: CFAR thresholds for each point.
- `aoa`: angle of arrival estimates.
- `map.npz`: aggregate of detected CFAR points into a map; see `roverp cfar_map`.
- `pointcloud.npz`: aggregate CFAR point cloud; see `roverp cfar_pointcloud`.

`_fusion`: sensor fusion alignment data.
- `indices.npz`: time-synchronized indices of data from different sensors; see `roverp align`.

`imu`: collected IMU data.
- `meta.json`: channel metadata.
- `acc`: linear acceleration.
- `avel`: angular velocity.
- `rot`: orientation (euler angles).
- `ts`: timestamps.

`lidar`: collected Lidar data.
- **NOTE**: all lidar data is **staggered** according to the ouster format, and must be [destaggered](https://static.ouster.dev/sdk-docs/reference/lidar-scan.html#staggering-and-destaggering).
- All Lidar data has shape `(height, width)` where `height` is the number of beams (32, 64, or 128) and `width` is the number of encoder counts per revolution (typically 512, 1024, or 2048).
- `meta.json`: channel metadata.
- `lidar.json`: Lidar sensor intrinsics as specified by [sensor metadata](https://static.ouster.dev/sdk-docs/python/examples/basics-sensor.html#obtaining-sensor-metadata) in the `ouster-sdk` (`SensorInfo`)
- `nir`: near-infrared photons (i.e. modern Lidars are infrared cameras).
- `rfl`: object Lidar reflectances.
- `rng`: object range.
- `ts`: timestamps.

`_lidar`: decompressed Lidar data (i.e. without the outer LZMA compression layer with a ratio around 2x).
- `meta.json`: channel metadata.
- `rng`: decompressed range; see `roverp decompress`.
- `ts`: timestamps.

`_nerfstudio`: video and poses in [Nerfstudio format](https://docs.nerf.studio/quickstart/data_conventions.html).
- See `roverp as_nerfstudio`.

`radar`: collected radar data.
- `meta.json`: channel metadata.
- `radar.json`: radar intrinsics (shape, doppler resolution, range resolution).
- `iq`: raw IQ data in `(slow time, tx, rx, fast time)` order.
- `ts`: timestamps.

`_report`: visualizations and other reporting.
- `data.mp4`: data visualization; see `roverp video`.
- `speed.pdf`: speed and postion trace; see `roverp report`.

`_radar`: processed radar data.
- `meta.json`: channel metadata.
- `pose.npz`: poses from the SLAM pipeline aligned to radar frame timestamps; see `roverp sensorpose -s radar`.
- `hybrid`: post-FFT data using the hybrid `min(fft(x), fft(hann(x)))`; see `roverp fft`.
- `sim_lidar`: lidar-based simulation; see `roverp simulate -m lidar`.
- `sim_cfar`: CFAR-based simulation; see `roverp simulate -m cfar`.
- `sim_nearest`: nearest-neighbor simulation; see `roverp nearest`.
- `ts`: timestamps.

`_scratch`: scratch space.
- `pose.bag`: output poses from `make trajectory` or `make lidar`.
- `lidar.bag`: Lidar and IMU data in rosbag format for cartographer; see `roverp rosbag`.

`_slam`: SLAM outputs.
- `lidar.bag_*`: outputs from Cartographer via `make lidar`.
- `lidar.pbstream`: outputs from Cartographer via `make trajectory` or `make lidar`.
- `trajectory.csv`: output trajectory via `make trajectory` or `make lidar`.
- `map.npz`: Lidar points as an occupancy grid; see `roverp lidarmap`.
