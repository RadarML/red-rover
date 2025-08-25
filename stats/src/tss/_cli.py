"""CLI interface for calculating statistics."""

from io import StringIO

import tyro
import yaml

from . import api


def _cli(
    path: str, /,
    pattern: str =  r"^(?P<experiment>(.*)).npz$",
    key: str = "loss", timestamps: str | None = None,
    experiments: list[str] | None = None,
    baseline: str | None = None,
    follow_symlinks: bool = False,
    cut: float | None = None,
    config: str | None = None,
) -> None:
    """Calculate statistics for time series metrics.

    - pipe `tss ... > results.csv` to save the results to a file
    - use `--config config.yaml` to avoid having to specify all these arguments

    Args:
        path: directory to find evaluations in.
        pattern: regex pattern to match the evaluation directories.
        key: name of the metric to load from the result files.
        timestamps: name of the timestamps to load from the result files.
        experiments: list of experiments to include in the results.
        baseline: baseline experiment for relative statistics.
        follow_symlinks: whether to follow symbolic links. May lead to infinite
            recursion if `True` and the `path` contains self-referential links!
        cut: cut each time series when there is a gap in the timestamps larger
            than this value if provided.
        config: load all of these values from a yaml configuration file
            instead.
    """
    if config is not None:
        with open(config) as f:
            cfg = yaml.safe_load(f)
            return _cli(
                path,
                pattern=cfg.get("pattern",  r"^(?P<experiment>(.*)).npz$"),
                experiments=cfg.get("experiments", None),
                key=cfg.get("key", "loss"),
                timestamps=cfg.get("timestamps", None),
                baseline=cfg.get("baseline", None),
                cut=cfg.get("cut", None),
                follow_symlinks=follow_symlinks)

    index = api.index(path, pattern=pattern, follow_symlinks=follow_symlinks)
    df = api.dataframe_from_index(
        index, key=key, baseline=baseline,
        experiments=experiments, cut=cut, timestamps=timestamps)

    buf = StringIO()
    df.to_csv(buf)
    print(buf.getvalue())


def cli_main() -> None:
    tyro.cli(_cli)
