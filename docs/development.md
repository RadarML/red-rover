# Development

While each red-rover module is intended to be installed and used independently, we provide a single development environment based on a shared `pyproject.toml`.

!!! warning

    While `roverc` (`red-rover/collect`) is nominally a part of the development environment (e.g., for documentation), it is not actually fully installed in the main environment. Developing `roverc` should use an actual physical data collection system, while the main environment can be used on any linux system.

## Setup

Assuming `uv` is [installed](https://docs.astral.sh/uv/getting-started/installation/), you can install all dependencies with `uv sync --all-extras` in `./red-rover`. This installs the documentation stack, testing / linting tools, and the `roverd: red-rover/format` and `roverp: red-rover/processing` modules as editable (i.e., `pip install -e`) links.

We also use [pre-commit hooks](https://pre-commit.com/); please install these with `uv run pre-commit install`.

!!! info

    You can also manually trigger the pre-commit hooks with `uv run pre-commit run`.

## Documentation

`red-rover` has a single documentation site using `mkdocs` / `mkdocs-material` / `mkdocstrings-python`. To build for development:

```sh
uv run --extra docs mkdocs serve
```

!!! info

    The documentation site is automatically built and deployed by GitHub Actions on push to `main`.

## Testing

Currently, only `roverd` comes with unit tests. Run with
```sh
export ROVERD_TEST_TRACE=/data/roverd/test-trace
uv run --all-extras pytest -ra --cov --cov-report=html --cov-report=term
```
where `ROVERD_TEST_TRACE` should point to a sample data trace with at least 0.5s of valid data across all modalities.

!!! tip

    Serve a live version of the code coverage report using
    ```sh
    uv run python -m http.server 8001 -d ./htmlcov
    ```
