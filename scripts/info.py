"""Get collected data metadata."""

import os
import json
import struct
import math


def _size(
    x: int, units: list = ['B', 'KB', 'MB', 'GB', 'TB'], suffix: str = ''
) -> str:
    if x < 1000:
        return "{:4.3g} {}".format(x, units[0]) + suffix
    else:
        return _size(x / 1000, units[1:]) + suffix


def _duration(path: str) -> float:
    with open(path, 'rb') as f:
        start, = struct.unpack('d', f.read(8))
        f.seek(-8, os.SEEK_END)
        end, = struct.unpack('d', f.read(8))
    return end - start


def report(path: str) -> dict:
    """Generate data collection report."""

    report = {"sensors": {}, "size": 0.0, "rate": 0.0}
    for sensor in os.listdir(path):
        with open(os.path.join(path, sensor, "meta.json")) as f:
            meta = json.load(f)

        n_samples = os.stat(os.path.join(path, sensor, "ts")).st_size // 8
        duration = _duration(os.path.join(path, sensor, "ts"))

        report["sensors"][sensor] = {
            "samples": n_samples, "duration": duration, "channels": {}}
        _channels = report["sensors"][sensor]["channels"]
        for channel, info in meta.items():
            ch_path = os.path.join(path, sensor, channel)
            actual_size = os.stat(ch_path).st_size
            actual_rate = actual_size / duration
            if info["format"] != "raw":
                raw_size = (
                    n_samples * math.prod(info["shape"])
                    * int(info["type"].lstrip("uifc")) // 8)
                _channels[channel] = {
                    "size": actual_size, "raw_size": raw_size,
                    "ratio": raw_size / actual_size, "rate": actual_rate}
            else:
                _channels[channel] = {"size": actual_size}

            report["size"] += actual_size
            report["rate"] += actual_rate

    return report


def _parse(p):
    p.add_argument("-p", "--path", help="Target path.")


def _main(args):
    rpt = report(args.path)

    print("total: {} ({})".format(
        _size(rpt["size"]), _size(rpt["rate"], suffix='/s')))
    for sensor, info in rpt["sensors"].items():
        print("{}: n={}, t={:.2f}".format(
            sensor, info["samples"], info["duration"]))
        for channel, ch_info in info["channels"].items():
            if "raw_size" in ch_info:
                print("    {:6}: {} (raw={}, ratio={:5.2f}, rate={})".format(
                    channel.split('.')[0], _size(ch_info["size"]),
                    _size(ch_info["raw_size"]), ch_info["ratio"],
                    _size(ch_info["rate"], suffix='/s')))
