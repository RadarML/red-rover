"""Decompress lidar data.

Inputs: lidar/*
Outputs: _lidar/*
"""

from rover import Dataset


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-c", "--channels", nargs='+',
        help="Channels to decompress.", default=["rng"])


def _main(args):

    ds = Dataset(args.path)
    
    lidar = ds["lidar"]
    _lidar = ds.create("_lidar", exist_ok=True)
    
    for k, v in lidar.config.items():
        v = dict(**v)
        v["format"] = "raw"
        if k == 'ts':
            _lidar.create(k, v).write(lidar[k].read())
        elif k in args.channels:
            print("Decompressing:", k)
            _lidar.create(k, v).consume(lidar[k].stream())
        else:
            print("Skipping:", k)
