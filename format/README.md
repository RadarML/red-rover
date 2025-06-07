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
