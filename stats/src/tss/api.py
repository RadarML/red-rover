"""High level API for loading and calculating statistics from experiments."""

import os
import re
from multiprocessing import pool
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeVar

import numpy as np
import optree
import pandas as pd
from jaxtyping import Float64, Num
from scipy.stats import norm

from .stats import NDStats
from .utils import cut_trace, intersect_difference

LeafType = TypeVar("LeafType", bound=np.ndarray)

if TYPE_CHECKING:
    # NOTE: mkdocstrings uses TYPE_CHECKING mode, so we put the docstring here.
    NestedValues = Sequence["NestedValues"] | LeafType
    """An arbitrarily nested sequence, parameterized by a leaf type.

    For example, these are valid examples of
    `NestedValues[Float[np.ndarray, "_N"]]`:
    ```python
    nested_leaf = Float[np.ndarray, "N1"]
    nested_list = [Float[np.ndarray, "N1"], Float[np.ndarray, "N2"]]
    nested_list_list = [
        [Float[np.ndarray, "N1"], Float[np.ndarray, "N2"]],
        [Float[np.ndarray, "N3"], Float[np.ndarray, "N4"]],
    ]
    ```
    """
else:
    NestedValues = Sequence[Any] | LeafType


def index(
    path: str, pattern: str | re.Pattern, follow_symlinks: bool = False
) -> dict[str | None, dict[str | None, str]]:
    r"""Recursively find all evaluations matching the given pattern.

    !!! tip

        LLM chat bots are very good at writing simple regex patterns!

    The pattern can have two groups: `experiment`, and `trace`, which
    respectively indicate the name of the experiment and trace. If either group
    is omitted, it is set as `None`.

    !!! example

        ```python
        # match `<experiment>/eval/<trace>.npz`
        index(path, r'^(?P<experiment>(.*))/eval/(?P<trace>(.*))\.npz$')
        # match `<experiment>.npz`
        index(path, r'^(?P<experiment>(.*))\.npz$')
        ```

    Args:
        path: directory to start searching from.
        pattern: regex pattern to match the evaluation directories.
        follow_symlinks: whether to follow symbolic links.

    Returns:
        A two-level dictionary, where the first level keys are the experiment
            names, the second level keys are the trace names, and the values
            are paths to the matching files.
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    manifest = {}

    def _find(path, base):
        matches = pattern.match(os.path.relpath(path, base))
        if matches is not None:
            groups = matches.groupdict()
            manifest.setdefault(
                groups.get('experiment', None), {}
            )[groups.get('trace', None)] = path
        elif os.path.isdir(path):
            if follow_symlinks or not os.path.islink(path):
                for p in os.listdir(path):
                    _find(os.path.join(path, p), base)

    _find(path, path)
    return manifest


def experiments_from_index(
    index: dict[str | None, dict[str | None, str]],
    key: str, timestamps: str | None = None,
    experiments: Sequence[str | None] | str | None = None,
    cut: float | None = None, workers: int = -1
) -> tuple[
    Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None,
    list[str]
]:
    """Load experiment results from indexed result files.

    Each results file is expected to be a `.npz` file containing metric and
    metadata arrays; the keys for these arrays should be specified by `key` and
    `timestamps`, respectively.

    - These arrays should all have the same leading axis length.
    - The metric array should have only a single axis.

    !!! warning

        Only sequences which are present in all experiments will be loaded.
        Check the returned `common` list to make sure it matchse what you
        expect!

    !!! tip

        A `timestamps` key can optionally be provided.

        - If not provided, the metrics are assumed to be at identical
            timestamps.
        - If multiple timestamps are present, the last one is used.

    Args:
        index: 2-level dictionary with experiment names, sequence/trace names,
            and paths to the result files; see [`index`][^.].
        key: name of the metric to load from the result files.
        timestamps: name of the timestamps to load from the result files.
        experiments: list of experiment names to load from the index (or a
            regex filter); loads all experiments if not specified.
        cut: cut each time series when there is a gap in the timestamps larger
            than this value if provided; see [`cut_trace`][tss.utils.].
        workers: number of worker threads to use when loading. If `<0`, load
            all in parallel; if `=0`, load all in the main thread.

    Returns:
        A dictionary of metric values (as a list of metric values by sequence).
        A dictionary of timestamps (or `None` if not specified).
        A list of the common sequence/trace names which correspond to the
            loaded metrics.
    """
    if len(index) == 0:
        raise ValueError("Could not fetch experiments: the index is empty.")

    if experiments is None:
        experiments = list(index.keys())
    elif isinstance(experiments, str):
        re_filter = re.compile(experiments)
        experiments = [x for x in index.keys() if re_filter.match(str(x))]
        if len(experiments) == 0:
            raise ValueError(
                f"No experiments found matching the filter: {experiments}")

    common = list(set.intersection(
        *[set(index[k].keys()) for k in experiments]))
    if workers < 0:
        workers = len(common) * len(experiments)

    def _load(path: str):
        data = np.load(path)
        if timestamps is not None:
            t = data[timestamps]
            t = t.reshape(t.shape[0], -1)[:, -1]
            if cut is not None:
                ytyt = cut_trace(t, (data[key], t), gap=cut)
                return list(zip(*ytyt))
            else:
                return [data[key]], [data[timestamps]]
        else:
            return [data[key]]

    iterload = [(x, s) for x in experiments for s in common]
    if workers == 0:
        loaded = [_load(index[x][s]) for x, s in iterload]
    else:
        with pool.ThreadPool(workers) as p:
            loaded = list(p.map(_load, [index[x][s] for x, s in iterload]))

    if timestamps is not None:
        yy, tt = {}, {}
        for (x, s), (y, t) in zip(iterload, loaded):
            yy.setdefault(x, []).extend(y)
            tt.setdefault(x, []).extend(t)
        return yy, tt, common
    else:
        yy = {}
        for (x, s), y in zip(iterload, loaded):
            yy.setdefault(x, []).extend(y)
        return yy, None, common


def stats_from_experiments(
    y: Mapping[str, NestedValues[Num[np.ndarray, "_N"]]],
    t: Mapping[str, NestedValues[Float64[np.ndarray, "_N"]]] | None = None,
    baseline: str | None = None, workers: int = -1
) -> tuple[list[str], NDStats, NDStats | None]:
    """Calculate statistics from experiment results.

    Args:
        y: mapping of experiment names and metric values.
        t: mapping of experiment names and timestamps. If not provided, the
            metrics are assumed to be at identical timestamps.
        baseline: baseline experiment for relative statistics.
        workers: number of worker threads to use for computation.

    Returns:
        Names of each experiment corresponding to leading axis in the output
            statistics.
        Absolute statistics for the provided metric.
        Relative statistics (difference relative to the specified baseline), if
            provided.
    """
    n_sorted = sorted(y.keys())
    if t is not None and set(n_sorted) != set(t.keys()):
        raise ValueError(
            f"Keys of `y` and `t` must match if `t` is provided: "
            f"y:{list(y.keys())}, t:{list(t.keys())}")

    y_sorted = [y[k] for k in n_sorted]
    stats_abs = NDStats.from_values(y_sorted, workers=workers)
    if baseline is not None:
        if t is not None:
            t_sorted = [t[k] for k in n_sorted]
            diff = optree.tree_map(
                intersect_difference,
                y_sorted, [y[baseline]] * len(y),  # type: ignore
                t_sorted, [t[baseline]] * len(y))  # type: ignore
            stats_rel = NDStats.from_values(
                diff, workers=workers)  # type: ignore
        else:
            diff = optree.tree_map(
                lambda x, y: x - y, y_sorted,   # type: ignore
                [y[baseline]] * len(y))  # type: ignore
            stats_rel = NDStats.from_values(
                diff, workers=workers)  # type: ignore
    else:
        stats_rel = None
    return n_sorted, stats_abs, stats_rel


def dataframe_from_stats(
    names: list[str], abs: NDStats, rel: NDStats | None = None,
    baseline: str | None = None
) -> pd.DataFrame:
    """Create a dataframe from (possibly un-aggregated) experiment statistics.

    Returns a dataframe where each row is a different experiment.

    - `abs/(mean|std|stderr|zscore|n|ess)`: absolute statistics for the
        provided metric for each experiment.
    - `rel/(mean|std|stderr|zscore|n|ess)`: relative statistics for the
        provided metric for each experiment, relative to the `baseline`. If no
        `baseline` is provided, these columns are not included.
    - `pct/(mean|stderr)`: percent difference and standard error relative to
        the `baseline`, computed as `100 * <rel/mean>/<abs/mean>` and
        `100 * <rel/stderr>/<abs/mean>`.

    Args:
        names: names of the experiments corresponding to the leading axis in
            the input statistics.
        abs: absolute statistics for the provided metric for each experiment.
        rel: optional relative statistics.
        baseline: name of the experiment used as the baseline.

    Returns:
        Dataframe with statistics for each experiment.
    """
    df = abs.reshape(
        len(names), -1).sum(axis=-1).as_df(names, prefix="abs/")

    if rel is not None and baseline is None:
        raise ValueError(
            "Provided relative statistics `rel`, but the `baseline` used is "
            "not specified.")

    if rel is not None:
        df_rel = rel.reshape(
            len(names), -1).sum(axis=-1).as_df(names, prefix="rel/")
        df = df.merge(df_rel, on='name')
        _baseline = df.loc[baseline]['abs/mean']
        df['pct/mean'] = df['rel/mean'] / _baseline * 100
        df['pct/stderr'] = df['rel/stderr'] / _baseline * 100

        z = norm.ppf(1 - 0.05 / 2 / (len(names) - 1))
        df['p0.05'] = (df['rel/mean'] / df['rel/stderr']) > z

    return df


def dataframe_from_index(
    index: dict[str | None, dict[str | None, str]],
    key: str, timestamps: str | None = None,
    experiments: Sequence[str | None] | None = None,
    cut: float | None = None, baseline: str | None = None, workers: int = -1
) -> pd.DataFrame:
    """Load and calculate statistics from indexed experiment results.

    See (1) [`dataframe_from_stats`][^.], (2) [`stats_from_experiments`][^.],
    and (3) and [`experiments_from_index`][^.].

    Args:
        index: 2-level dictionary with experiment names, sequence/trace names,
            and paths to the result files; see [`index`][tss.api.].
        key: name of the metric to load from the result files.
        timestamps: name of the timestamps to load from the result files.
        experiments: list of experiment names to load from the index; loads all
            experiments if not specified.
        cut: cut each time series when there is a gap in the timestamps larger
            than this value if provided; see [`cut_trace`][tss.utils.].
        baseline: baseline experiment for relative statistics.
        workers: number of worker threads to use when loading. If `<0`, load
            all in parallel; if `=0`, load all in the main thread.

    Returns:
        Dataframe with statistics for each experiment.
    """
    y, t, _ = experiments_from_index(
        index, key, timestamps=timestamps, experiments=experiments,
        cut=cut, workers=workers)
    names, stats_abs, stats_rel = stats_from_experiments(
        y, t, baseline=baseline, workers=workers)
    df = dataframe_from_stats(names, stats_abs, stats_rel, baseline=baseline)
    return df
