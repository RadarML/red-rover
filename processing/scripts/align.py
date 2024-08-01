"""Get sensor timestamp alignments as indices.

Inputs:
    - any set of sensors
    - `_{sensor}/pose.npz` if `--require_pose` is set

Outputs:
    - `_fusion/indices.npz` unless overridden.

Keys:
    - `indices`: corresponding `u32` data indices of each of the specified
      `--sensors` in the provided order, with the sensor in axis 0 and
      time index in axis 1.
    - `sensors`: string array recording the provided `--sensors`.
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
    p.add_argument(
        "-r", "--require_pose", default=False, action='store_true',
        help="Require valid poses for inclusion.")


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

    if args.require_pose:
        _start = []
        _end = []
        for s, t in zip(args.sensors, timestamps):
            mask = np.load(
                os.path.join(args.path, '_' + s, "pose.npz"))["mask"]
            _start.append(t[np.argmax(mask)])
            _end.append(t[-np.argmax(mask[::-1])])
        start = max(_start)
        end = min(_end)
    else:
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
