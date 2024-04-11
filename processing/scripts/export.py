"""Export dataset to another location.

Inputs: an entire dataset.
Outputs: the dataset is copied into the `dst` directory (e.g. on an external
    drive or file server), except for easily recreated processed output files.
"""

import os
import shutil
from tqdm import tqdm


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset to copy.")
    p.add_argument("-o", "--out", help="Destination directory.")
    p.add_argument(
        "--metadata", default=False, action='store_true',
        help="Copy metadata only.")


def _main(args):

    def should_copy(path):
        if args.metadata:
            return (
                (not path.startswith('_')) and (
                    os.path.splitext(path)[1] in {'.json', '.yaml'}
                    or os.path.split(path)[-1] == 'ts'))
        else:
            if path.startswith('_') and "_report" not in path:
                return os.path.splitext(path)[1] in {'.npz', '.csv'}
            else:
                return True

    pairs = []
    for root, _, files in os.walk(args.src):
        for file in files:
            abspath = os.path.join(root, file)
            relpath = os.path.relpath(abspath, args.src)
            if should_copy(relpath):
                pairs.append((
                    os.stat(abspath).st_size, abspath,
                    os.path.join(args.dst, relpath)))

    pairs.sort(key=lambda x: x[0])
    totalsize = sum([s for s, _, _ in pairs])
    pbar = tqdm(total=totalsize, unit_scale=True, unit='B')
    for (size, src, dst) in pairs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        pbar.update(size)
