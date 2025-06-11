"""Data marshalling utilities."""

from typing import TypeVar

import numpy as np
import optree
from jaxtyping import Float, Num

LeafType = TypeVar("LeafType")
MetricTree = dict[str, "MetricTree"] | LeafType


def tree_flatten(
    tree: MetricTree[LeafType], _path: list[str] = []
) -> dict[str, LeafType]:
    """Flatten a nested structure into a dictionary with path-like keys.

    Inverse of [`tree_unflatten`][..].

    Args:
        tree: input tree structure. Must consist only of dictionaries, where
            each leaf value is a numpy array with the same leading batch size.

    Returns:
        A dictionary, where keys correspond to `/`-joined strings of the key
            paths to each leaf.
    """
    if isinstance(tree, dict):
        out = {}
        for key, value in tree.items():
            out.update(tree_flatten(value, _path=_path + [key]))
        return out
    else:
        return {"/".join(_path): tree}


def tree_unflatten(flattened: dict[str, LeafType]) -> MetricTree[LeafType]:
    """Unflatten a dictionary with path-like keys into a nested structure.

    Inverse of [`tree_flatten`][..].

    Args:
        flattened: input dictionary, where keys are `/`-joined strings of the
            key paths to each leaf.

    Returns:
        Expanded dictionary with arbitrary nesting.
    """
    tree = {}
    for key, value in flattened.items():
        fullpath = key.split("/")
        current = tree
        for subpath in fullpath[:-1]:
            current = current.setdefault(subpath, {})
        current[fullpath[-1]] = value
    return tree


TValue = TypeVar("TValue")

def cut_trace(
    timestamps: Float[np.ndarray, "N"],
    values: TValue,
    gap: float = 20.
) -> list[TValue]:
    """Cut trace into multiple sub-traces based on gaps in timestamps.

    !!! tip

        Set `gap` to a duration, in seconds, which is greater than (but of
        similar magnitude) than the expected effective sampling rate of the
        time series.

    Args:
        timestamps: measurement timestamps.
        values: time series measurement values. Is expected to be a
            `PyTree[Shaped[np.ndarray, "N ..."]]`, where the length of each
            node is equal to the number of `timestamps` passed.
        gap: timestamp gap, nominally in seconds, which denotes a new trace.

    Returns:
        A list of metrics for each sub-trace, with the same structure and type
            as the inputs.
    """
    cuts, = np.where(np.diff(timestamps) > gap)
    cuts = np.concatenate([[0], cuts, [timestamps.shape[0]]])

    return [
        optree.tree_map(lambda x: x[start:stop], values)  # type: ignore
        for start, stop in zip(cuts[:-1], cuts[1:])]


def intersect_difference(
    y1: Num[np.ndarray, "N1"], y2: Num[np.ndarray, "N2"],
    t1: Num[np.ndarray, "N1"] | None = None,
    t2: Num[np.ndarray, "N2"] | None = None
) -> Num[np.ndarray, "N"]:
    """Compute the difference between two time series at common timestamps.

    !!! info

        If `t1` and `t2` are not provided, the two time series are assumed to
        be synchronized.

    Args:
        y1: first time series.
        y2: second time series.
        t1: timestamps for the first time series.
        t2: timestamps for the second time series.

    Returns:
        Differences `y1 - y2` at common timestamps, sorted in increasing order.
    """
    if t1 is None or t2 is None:
        if y1.shape != y2.shape:
            raise ValueError(
                f"If no timestamps are provided, both time series must have "
                f"the same shape: y1:{y1.shape} != y2:{y2.shape}.")
        return y1 - y2

    _, i1, i2 = np.intersect1d(t1, t2, return_indices=True)
    return y1[i1] - y2[i2]
