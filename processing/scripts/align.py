"""Get sensor timestamp alignments as indices.

Inputs: any set of sensors
Outputs: `_fusion/indices.npz` unless overridden.
"""

import os
import time
import numpy as np

from rover import Dataset


def _parse(p):
    p.add_argument("-p", "--path", help="Dataset path.")
    p.add_argument(
        "-o", "--out", default=None,
        help="Output dataset; defaults to the input dataset.")
    p.add_argument(
        "-s", "--sensors", nargs='+', default=['lidar', 'radar'],
        help="Sensors to align.")
    p.add_argument(
        "-m", "--mode", default="left", help="Alignment mode. `left`: align "
        "to first sensor; `union`: align to all sensors.")


def _main(args):

    if args.out is None:
        args.out = args.path
    out = os.path.join(args.out, "_fusion", "indices.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    ds = Dataset(args.path)
    timestamps = [ds.get(s).timestamps(smooth=False) for s in args.sensors]

    if args.mode == 'left':
        ref = timestamps[0]
    elif args.mode == 'union':
        ref = np.sort(np.concatenate(timestamps))
    else:
        print(f"Unknown mode: {args.mode}.")
        exit(1)

    t0 = time.time()

    start = max(t[1] for t in timestamps)
    end = min(t[-3] for t in timestamps)
    ref = ref[(ref > start) & (ref < end)]

    def advance(target, start, ts):
        i = 0
        for i in range(start, len(ts) - 1):
            if ts[i] <= target and target <= ts[i + 1]:
                break
        else:
            breakpoint()

        nearest = target - ts[i] < ts[i + 1] - target
        return i, i if nearest else i + 1

    res = np.zeros((len(ref), len(timestamps)), dtype=np.uint32)
    j = [0 for _ in timestamps]
    for i, t in enumerate(ref):
        j, aligned = list(zip(*[
            advance(t, start, ts) for start, ts in zip(j, timestamps)]))
        res[i] = np.array(aligned)

    print(f"Aligned {len(res)} indices in {time.time() - t0:.3f}s.")
    np.savez(out, indices=res, sensors=np.array(args.sensors, dtype=str))
