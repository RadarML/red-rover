## Install

Install directly from github with:
```sh
pip install "roverd[video,ouster]@git+ssh://git@github.com/WiseLabCMU/red-rover.git#subdirectory=format"
```
- `roverd[video,ouster]`: install with video reading (`opencv-python-headless`) and lidar metadata (`ouster-sdk`) dependencies. If these features are not required, these extras can be omitted.
- The package is fetched directly from our data collection system mono-repo at `github.com/WiseLabCMU/red-rover.git` in this subdirectory (`subdirectory=format`)

## Local Development

If this repository is cloned locally, you can install it with
```sh
pip install -e .[video,ouster]
```
- **NOTE**: when using `pip install` from local files, you can replace `.` with the relative path to the `red-rover/format` directory, e.g. `pip install ./red-rover/format[video,ouster]` from the root `RadarML` directory.

To run unit tests:
```
python -m unittest discover
```

## Design Goals

The rover dataset format, `roverd` is designed to handle data collection and storage for concurrent sensor streams with a fixed dimension, such as camera, lidar, or radar data. Its design has the following goals:

**Appendable**: the data format should be easy to append new observations to.

> By writing files in a simple binary format in a time-synchronous way within each sensor stream, appending data is trivial -- just append data to the file; no metadata updates are required.

**Fail-safe**: if the data collection program or device were to fail during data collection, the amount of lost or corrupted data should be minimized.

> Since data is simply appended without any summary metadata which needs to be committed when the file is closed, program or device crashes only lose any data which is currently buffered.

**Readable with common tools**: the collected data and metadata should be readable with off-the-shelf software (e.g. what ships with a stock linux, windows, or Mac OS distribution) to the greatest extent possible.

> Data is organized using the native file system, and can be browsed using any file manager. Metadata files are simple `.json` or `.yaml` files, which can be readily viewed or modified with generic text editors.

**Extendable**: it should be easy to use arbitrary file formats, possibly with domain-specific compression techniques.

> Data channels can specify an arbitrary file format, which is redirected to the appropriate reader or writer.

**Simple to implement**: it should be easy and simple to implement any subset of functionality for operating on datasets.

> Roverd is incredibly simple, and users only need to implement the parts of the specification that they need.

### Why not HDF5?

HDF5 is [commonly criticized](https://cyrille.rossant.net/moving-away-hdf5/) for its clunkiness: in short, HDF5 is a monster of a standard, and our use cases require maybe 1% of its functionality. By using our own file format, we enable users to view and edit metadata files using common GUI and command line tools (e.g. file managers and text editors), while vastly simplifying the system by using the native file system instead of HDF5's filesystem-within-a-filesystem.

If we were to use HDF5, we would also run into trouble when using domain-specific compression formats, e.g. video:
- If included the video inside the HDF5 file, off-the-shelf video players would not be able to open it.
- Alternatively, we could store the video externally; however, then we would need to implement our own metadata management for these external files and a wrapper to integrate the two, which is almost as much work as spinning a custom dataset format!

### Why not npz or npy?

NPZ or NPY files cannot be easily appended to, since they require metadata to be written "up front." We also would need to implement a parallel data management system anyways for videos and other domain-specific formats, and would not be able to rely solely on the array naming system in `npz` files.


## Specifications

### Channel

A `Channel` (e.g. `iq`, `acc`, `rng`) is a single sequential, homogenous data stream from a sensor.
- Each channel is stored as a file, and may be stored in a raw binary format, compressed, or any other format (e.g. `raw`, `lzma`, `mjpeg`).
    - Channel names may have file extensions (e.g. `.avi`) if the file format can be interpreted by off-the-shelf programs.
    - The "shape" of each channel is described by the constant axes of the data, i.e. the overall array for a channel with `N` observations has shape `(N, *shape)`.
    - All data should be stored in little endian.
- Channel names cannot be `meta.json` or `ts`.
    - ... with exception of the special "timestamp" channel, which should have format `raw` and type `f8` (8-byte double precision float), and contain the epoch time, measured in seconds.

### Sensor

A `Sensor` (e.g. `imu`, `lidar`, `radar`, `camera`) is a collection of time-synchronous channels, where sequential entries in each channel correspond to a single series of timestamps.
- Sensors are represented by directories, containing one file for each channel.
    - Sensor names must not start with an underscore (`_`).
    - Sensor directories may contain additional non-channel metadata / sensor intrinsics files other than `meta.json`, e.g. `lidar.json`.
- Each sensor directory must have a `meta.json` file, which includes the data type and file format for each sensor channel, for example:
    ```json
    {
        "iq": {
            "format": "raw", "type": "i2", "shape": [64, 3, 4, 512],
            "desc": "Raw I/Q stream."
        },
        "ts": {
            "format": "raw", "type": "f8", "shape": [],
            "description": "Timestamp, in seconds."
        }
    }
    ```
    - Data types are specified according to the [numpy size-in-bytes convention](https://numpy.org/doc/stable/reference/arrays.dtypes.html), e.g. `i2` for 2-byte signed integer (`short`, `int16`) or `f8` for 8-byte floating point (`double`, `float64`).


### Dataset

A `Dataset` is a directory containing a collection of (likely asynchronous) sensors.
- Any directories inside the dataset directory containing a `meta.json` file are treated as sensors.
    - A `README`, `config.yaml`, or other metadata files can be placed in the root dataset folder if desired.
    - Any files which do not contain raw collected data should be placed in directories starting with an underscore (e.g. `_scratch`, `_out`, `_radad`).
    - Processed output files placed in a directory along with a valid `meta.json` are interpreted as "virtual" sensors, e.g. decompressed or processed data.
- Example structure:
    ```
    example/
        _scratch/       # ignored
            ...
        _odom           # virtual sensor
            acc
            avel
            meta.json
            rot
            ts
        camera/         # physical sensor
            meta.json
            ts
            vdeo.avi
        ...
        config.yaml     # dataset metadata
        ...
    ```
