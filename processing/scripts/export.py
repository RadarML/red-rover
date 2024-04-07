"""Export dataset to another location.

Inputs: an entire dataset.
Outputs: the dataset is copied into the `dst` directory (e.g. on an external
    drive or file server), except for easily recreated processed output files.
"""

import os
import shutil
from tqdm import tqdm


def _parse(p):
    p.add_argument("src", help="Dataset to copy.")
    p.add_argument("dst", help="Destination directory.")


def _main(args):

    def should_copy(path):
        if path.startswith('_') and "_report" not in path:
            if path.endswith(".npz") or path.endswith(".csv"):
                return True
            else:
                return False
        else:
            return True

    out = os.path.join(args.dst, os.path.basename(os.path.normpath(args.src)))

    pairs = []
    for root, _, files in os.walk(args.src):
        for file in files:
            abspath = os.path.join(root, file)
            relpath = os.path.relpath(abspath, args.src)
            if should_copy(relpath):
                pairs.append((
                    os.stat(abspath).st_size, abspath,
                    os.path.join(out, relpath)))

    pairs.sort(key=lambda x: x[0])
    totalsize = sum([s for s, _, _ in pairs])
    pbar = tqdm(total=totalsize, unit_scale=True, unit='B')
    for (size, src, dst) in pairs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        pbar.update(size)
