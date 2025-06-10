"""Time series metric statistics."""

from jaxtyping import install_import_hook

with install_import_hook("tsms", "beartype.beartype"):
    from .ablations import dataframe_from_ablations, stats_from_ablations
    from .stats import NDStats, autocorrelation, effective_sample_size
    from .utils import (
        cut_trace,
        intersect_difference,
        tree_flatten,
        tree_unflatten,
    )


__all__ = [
    "stats_from_ablations", "dataframe_from_ablations",
    "NDStats", "autocorrelation", "effective_sample_size",
    "tree_flatten", "tree_unflatten", "cut_trace", "intersect_difference"
]
