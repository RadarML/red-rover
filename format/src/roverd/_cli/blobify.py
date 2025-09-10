"""Convert default `roverd` channels to blob channels."""

import os
import shutil
import subprocess

from roverd import Trace
from roverd.sensors import DynamicSensor


def cli_blobify(src: str, dst: str, /, workers: int = 64) -> None:
    """Convert a trace to use blob channels.

    ```sh
    uv run roverd blobify /path/to/src /path/to/dst
    ```

    !!! warning

        This script requires `ffmpeg` to be installed, which we use as a fast
        way to split `mjpeg` video files into individual frames.

    !!! danger

        Blob channels store each sample as a separate file, which can lead to
        a very large number of files and high file system overhead. This should
        only done on a proper blob storage backend, e.g. AWS S3 or Azure Blob
        Storage! Local file systems or HPC storage systems should stick to
        the default channel types.

    Args:
        src: path to the source trace.
        dst: path to the output trace.
        workers: number of worker threads for writing blobs.
    """
    def _copy(file: str) -> None:
        shutil.copy(os.path.join(src, file), os.path.join(dst, file))

    trace = Trace.from_config(src, sensors={
        "_camera": None, "radar": None, "lidar": None,
        "camera": None, "imu": None,
    })

    os.makedirs(dst, exist_ok=True)
    for name, sensor in trace.sensors.items():
        assert isinstance(sensor, DynamicSensor)
        s_copy = DynamicSensor(
            os.path.join(dst, name), create=True, exist_ok=True)

        for ch_name, channel in sensor.channels.items():
            if ch_name in {"ts", "valid", "rot", "acc", "avel"}:
                ch_copy = s_copy.create(ch_name, sensor.config[ch_name])
                ch_copy.write(channel.read(start=0, samples=-1))
            else:
                cfg = {
                    "type": sensor.config[ch_name]['type'],
                    "shape": channel.shape
                }
                if ch_name == 'video.avi':
                    cfg["format"] = "jpg"
                    ch_copy = s_copy.create(ch_name, cfg)
                    os.makedirs(ch_copy.path, exist_ok=True)
                    subprocess.call(
                        f"ffmpeg -i {channel.path} -c:v copy -f image2 "
                        f"{ch_copy.path}/%06d.jpg", shell=True)
                elif ch_name == 'iq':
                    cfg["format"] = "npz"
                    ch_copy = s_copy.create(ch_name, cfg, args={
                        "compress": False, "workers": workers})
                    ch_copy.write(channel.read())
                else:
                    cfg["format"] = "npz"
                    ch_copy = s_copy.create(ch_name, cfg, args={
                        "compress": True, "workers": workers})
                    ch_copy.write(channel.read())

    _copy("lidar/lidar.json")
    _copy("radar/radar.json")
    os.makedirs(os.path.join(dst, "_radar"), exist_ok=True)
    _copy("_radar/pose.npz")
    _copy("config.yaml")
