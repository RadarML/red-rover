"""Time series metric statistics.

The high level API is broken up into four steps:

1. [`index`][.]: Index evaluation files using a regex pattern.

    !!! info

        If you are only interested in a subset of evaluation traces, you can
        filter them at this stage.

2. [`experiments_from_index`][.]: Load data for each indexed result file, or
    a subset of experiments.

3. [`stats_from_experiments`][.]: Compute statistics for each experiment. See
    [`NDStats`][.stats.] and [`effective_sample_size`][.stats.] for more
    details about how and what statistics are computed.

4. [`dataframe_from_stats`][.]: Aggregate the statistics into a readable
    dataframe, ready to be plotted or exported.

!!! tip

    We also provide [`dataframe_from_index`][.], which combines the last three
    steps into a single function for convenience.
"""

from jaxtyping import install_import_hook

with install_import_hook("tsms", "beartype.beartype"):
    from . import stats, utils
    from .api import (
        NestedValues,
        dataframe_from_index,
        dataframe_from_stats,
        experiments_from_index,
        index,
        stats_from_experiments,
    )


__all__ = [
    "NestedValues",
    "dataframe_from_index", "dataframe_from_stats",
    "experiments_from_index", "stats_from_experiments",
    "index", "stats", "utils"
]
