"""List datasets in a directory."""

import os

from roverd import Dataset


def cli_list(path: str, /, follow_symlinks: bool = False) -> None:
    """List traces (recursively) in a directory by looking for `config.yaml`.

    ```sh
    $ uv run roverd list /path/to/datasets
    example_a/trace1
    example_a/trace2
    example_b/trace1
    ...
    ```

    !!! tip

        This CLI is intended to be piped to other commands (and `roverd` CLI
        tools), e.g.:
        ```sh
        # Count traces
        uv run roverd list /path/to/traces | wc -l
        # Loop over traces
        for trace in `uv run roverd list /path/to/traces`; do echo $trace; done
        ```

    Args:
        path: directory to search inside.
        follow_symlinks: whether to follow symlinks when searching.
    """
    traces = Dataset.find_traces(path, follow_symlinks=follow_symlinks)
    relative_traces = [os.path.relpath(trace, path) for trace in traces]
    print('\n'.join(relative_traces))
