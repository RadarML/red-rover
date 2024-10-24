r"""Rover data processing pipeline.
::
     ______  _____  _    _ _______  ______
    |_____/ |     |  \  /  |______ |_____/
    |    \_ |_____|   \/   |______ |    \_
    Radar sensor fusion research platform.

Scripts are categorized as follows:

- Convert: convert data format for compatibility with external software.
- Create: generate a specific representation based on the input data.
- Export: convert or copy data for training or other distribution.
- Get: print or visualize dataset metadata.
- Render: create a video visualization of the data.
- Run: a compute (GPU) intensive operation to apply a given algorithm.
"""  # noqa: D205

_scripts = sorted([
    "rosbag", "sensorpose", "fft", "as_rover1", "lidarmap",
    "report", "video", "simulate", "compare", "compare2",
    "nearest", "slice", "export",
    "align", "cfar", "cfar2", "cfar_pointcloud", "cfar_map",
    "as_nerfstudio", "clip", "segment"
])
