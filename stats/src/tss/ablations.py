"""Statistics from ablation sets."""

from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeVar

import numpy as np
import optree
import pandas as pd
from jaxtyping import Float64, Num
from scipy.stats import norm

from .stats import NDStats
from .utils import intersect_difference

LeafType = TypeVar("LeafType", bound=np.ndarray)

if TYPE_CHECKING:
    NestedValues = Sequence["NestedValues"] | LeafType
else:
    NestedValues = Sequence[Any] | LeafType


def stats_from_ablations(
    y: Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    t: Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None = None,
    baseline: str | None = None
) -> tuple[NDStats, NDStats | None]:
    """Calculate statistics from ablation results.

    Args:
        y: mapping of ablation names and metric values.
        t: mapping of ablation names and timestamps. If not provided, the
            metrics are assumed to be at identical timestamps.
        baseline: baseline ablation for relative statistics.

    Returns:
        Tuple with absolute and relative statistics.
    """
    n_sorted = sorted(y.keys())
    if t is not None and set(n_sorted) != set(t.keys()):
        raise ValueError(
            f"Keys of `y` and `t` must match if `t` is provided: "
            f"y:{list(y.keys())}, t:{list(t.keys())}")

    y_sorted = [y[k] for k in n_sorted]
    stats_abs = NDStats.from_values(y_sorted)
    if baseline is not None:
        if t is not None:
            t_sorted = [t[k] for k in n_sorted]
            diff = optree.tree_map(
                intersect_difference,
                y_sorted, [y[baseline]] * len(y),  # type: ignore
                t_sorted, [t[baseline]] * len(y))  # type: ignore
            stats_rel = NDStats.from_values(diff)  # type: ignore
        else:
            diff = optree.tree_map(
                lambda x, y: x - y, y_sorted,   # type: ignore
                [y[baseline]] * len(y))  # type: ignore
            stats_rel = tsms.NDStats.from_values(diff)  # type: ignore
    else:
        stats_rel = None
    return stats_abs, stats_rel


def dataframe_from_ablations(
    y: Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    t: Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None = None,
    baseline: str | None = None
) -> pd.DataFrame:
    """Calculate aggregate statistics from ablation results.

    Returns a dataframe where each row is a different ablation.

    - `abs/(mean|std|stderr|zscore|n|ess)`: absolute statistics for the
        provided metric for each ablation.
    - `rel/(mean|std|stderr|zscore|n|ess)`: relative statistics for the
        provided metric for each ablation, relative to the `baseline`. If no
        `baseline` is provided, these columns are not included.
    - `pct/(mean|stderr)`: percent difference and standard error relative to
        the `baseline`, computed as `100 * <rel/mean>/<abs/mean>` and
        `100 * <rel/stderr>/<abs/mean>`.

    Args:
        y: mapping of ablation names and metric values.
        t: mapping of ablation names and timestamps. If not provided, the
            metrics are assumed to be at identical timestamps.
        baseline: baseline ablation for relative statistics.

    Returns:
        Dataframe with statistics for each ablation.
    """
    names = sorted(y.keys())
    stats_abs, stats_rel = stats_from_ablations(y, t=t, baseline=baseline)

    df = stats_abs.reshape(
        len(y), -1).sum(axis=-1).as_df(names, prefix="abs/")

    if stats_rel is not None:
        df_rel = stats_rel.reshape(
            len(y), -1).sum(axis=-1).as_df(names, prefix="rel/")
        df = df.merge(df_rel, on='name')
        _baseline = df.loc[baseline]['abs/mean']
        df['pct/mean'] = df['rel/mean'] / _baseline * 100
        df['pct/stderr'] = df['rel/stderr'] / _baseline * 100

        z = norm.ppf(1 - 0.05 / 2 / (len(y) - 1))
        df['p0.05'] = (df['rel/mean'] / df['rel/stderr']) > z

    return df
