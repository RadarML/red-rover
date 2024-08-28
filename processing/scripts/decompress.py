"""Export decompressed lidar data.

Inputs:
    - `lidar/*`

Outputs:
    - `_lidar/*`, unless overridden.

Channels:
    - `ts`: copy of original `ts` timestamp channel.
    - `rng`, `rfl`, `nir`: decompressed versions of the corresponding input
      channels. Note that destaggering, etc is not performed.
"""

from roverd import Dataset


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out",
        help="Output dataset path; should be a clone of the first dataset.")
    p.add_argument(
        "-c", "--channels", nargs='+',
        help="Channels to decompress.", default=["rng"])


def _main(args):

    ds = Dataset(args.path)

    if args.out is None:
        out = ds
    else:
        out = Dataset(args.out)

    lidar = ds["lidar"]
    _lidar = out.create("_lidar", exist_ok=True)

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
