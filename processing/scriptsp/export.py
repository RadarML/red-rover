"""Export dataset to another location.

The dataset is copied into the `--out` directory (e.g. on an external drive or
file server), except for easily recreated processed output files.
This includes:

- Any original collected data (i.e. not starting with `_`).
- Any data ending in `.npz` or `.csv`.

Inputs:
    - An entire dataset.

Outputs:
    - The specified `--out` directory.
"""

import os
import re
import shutil
from tqdm import tqdm


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset to copy.")
    p.add_argument("-o", "--out", help="Destination directory.")
    p.add_argument(
        "--metadata", default=False, action='store_true',
        help="Copy metadata only.")
    p.add_argument(
        "-t", "--test", default=False, action='store_true',
        help="Test run only (list files; don't actually copy)")


RE = re.compile(r"""^(
    [^_](.*) |                     # original data (doesn't start with '#')
    _camera/clip |                 # clip embeddings
    _(.*).(json|yaml|csv|npz) |    # metadata, generated data formats
    _(.*)/_report/(.*)             # any reports
)$""", re.VERBOSE)

RE_METADATA = re.compile(r"""^(
    (.*).(json|yaml) |             # pure metadata files
    (.*)/ts                        # timestamps
)$""", re.VERBOSE)


def should_copy(path: str, metadata: bool = False) -> bool:
    """Check if the file should be copied by referencing the above regex."""
    return bool(re.match(RE_METADATA if metadata else RE, path))


def _main(args):
    if args.test:
        args.out = ""

    pairs = []
    for root, _, files in os.walk(args.path):
        for file in files:
            abspath = os.path.join(root, file)
            relpath = os.path.relpath(abspath, args.path)
            if should_copy(relpath, metadata=args.metadata):
                dst = os.path.join(args.out, relpath)
                if not os.path.exists(dst):
                    pairs.append((os.stat(abspath).st_size, abspath, dst))

    if args.test:
        print("\n".join([x[-1] for x in pairs]))
        exit(0)

    pairs.sort(key=lambda x: x[0])
    totalsize = sum([s for s, _, _ in pairs])
    pbar = tqdm(total=totalsize, unit_scale=True, unit='B')
    for (size, src, dst) in pairs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        pbar.update(size)
