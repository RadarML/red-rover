# Dataset Format

## Usage

Dataset are created by `roverc`, and can be copied to a faster computer for processing. After processing, any created files should be stored in directories starting with `_`. These files do not need to be distributed, and can be excluded, e.g.
```sh
tar --exclude '*/_*' -cvf data.tar data
```

**NOTE**: the underlying data representations should already be compressed; further compressing the dataset as a whole is unlikely to yield a nontrivial compression ratio.

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
            "format": "raw", "type": "i16", "shape": [64, 3, 4, 512],
            "desc": "Raw I/Q stream."
        },
        "valid": {
            "format": "raw", "type": "u8", "shape": [],
            "desc": "True if this frame is complete (no zero-fill)."
        },
        "ts": {
            "format": "raw", "type": "f64", "shape": [],
            "description": "Timestamp, in seconds."
        }
    }
    ```

**Dataset**: a directory containing a collection of sensors.
- A copy of the active `config.yaml` is placed in the root dataset folder.
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
        config.yaml
    ```
