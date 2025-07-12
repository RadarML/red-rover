"""Get dataset metadata.

Only metadata for originally collected sensors is calculated.
"""

from typing import cast

from roverd import Trace, sensors


def _size(x, units=[' B', 'KB', 'MB', 'GB', 'TB'], suffix=''):
    if x < 1000:
        return "{:4.3g} {}".format(x, units[0]) + suffix
    else:
        return _size(x / 1000, units[1:]) + suffix


def cli_info(path: str, /,) -> None:
    """Print trace metadata.

    Only metadata for non-virtual (originally collected) sensors is shown.

    ```sh
    uv run roverd info /data/grt/bike/point.out
    ```

    ??? quote "Sample output"

        ```
        $ roverd info /data/grt/bike/point.out
        total    61.2 GB (32.8 MB/s)
        radar    29.3 GB (15.7 MB/s, n=37320, t=1865.9s)
            ts        299 KB (rate= 160  B/s)
            iq       29.3 GB (rate=15.7 MB/s)
            valid    37.3 KB (rate=  20  B/s)
        camera   21.1 GB (11.3 MB/s, n=56053, t=1868.3s)
            ts        448 KB (rate= 240  B/s)
            video    21.1 GB (raw= 349 GB, ratio=16.52, rate=11.3 MB/s)
        lidar    10.8 GB (5.78 MB/s, n=18570, t=1867.1s)
            ts        149 KB (rate=79.6  B/s)
            rfl      2.03 GB (raw=4.87 GB, ratio= 2.39, rate=1.09 MB/s)
            nir      5.27 GB (raw=9.74 GB, ratio= 1.85, rate=2.82 MB/s)
            rng      3.48 GB (raw=9.74 GB, ratio= 2.80, rate=1.86 MB/s)
        imu      8.22 MB ( 4.4 KB/s, n=186847, t=1868.3s)
            ts       1.49 MB (rate= 800  B/s)
            rot      2.24 MB (rate= 1.2 KB/s)
            acc      2.24 MB (rate= 1.2 KB/s)
            avel     2.24 MB (rate= 1.2 KB/s)
        ```

    Args:
        path: path to the trace directory.
    """
    ds = Trace.from_config(path)

    print("total    {} ({})".format(
        _size(ds.filesize),
        _size(ds.datarate, suffix='/s')))
    for sname, sensor in ds.sensors.items():
        sensor = cast(sensors.DynamicSensor, sensor)
        print("{:8} {} ({}, n={}, t={:.1f}s)".format(
            sname,
            _size(sensor.filesize),
            _size(sensor.datarate, suffix='/s'),
            len(sensor),
            sensor.duration))
        for cname, channel in sensor.channels.items():
            if sensor.config[cname]['format'] != 'raw':
                raw = channel.size * len(sensor)
                print("    {:8} {} (raw={}, ratio={:5.2f}, rate={})".format(
                    cname.split('.')[0],
                    _size(channel.filesize),
                    _size(raw),
                    raw / channel.filesize,
                    _size(channel.filesize / sensor.duration, suffix='/s')))
            else:
                print("    {:8} {} (rate={})".format(
                    cname.split('.')[0],
                    _size(channel.filesize),
                    _size(channel.filesize / sensor.duration, suffix='/s')))
