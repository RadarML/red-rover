"""Get dataset metadata.

Only metadata for originally collected sensors is calculated.
"""

from ..dataset import Dataset


def _size(x, units=[' B', 'KB', 'MB', 'GB', 'TB'], suffix=''):
    if x < 1000:
        return "{:4.3g} {}".format(x, units[0]) + suffix
    else:
        return _size(x / 1000, units[1:]) + suffix


def _parse(p):
    p.add_argument("-p", "--path", help="Target path.")


def _main(args):
    ds = Dataset(args.path)

    print("total    {} ({})".format(
        _size(ds.filesize),
        _size(ds.datarate, suffix='/s')))
    for sname, sensor in ds.sensors.items():
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
