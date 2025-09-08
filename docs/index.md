# **Red Rover**: A Multimodal mmWave Radar Spectrum Ecosystem

The **red rover**[^1] project is an end-to-end system for collecting, loading, and processing mmWave radar time signal data along with lidar and camera data designed for both [DART](https://wiselabcmu.github.io/dart/)-style reconstruction and [GRT](https://wiselabcmu.github.io/grt/)-style deep learning.

!!! info

    While packaged together as a mono-repo, each module is designed to be able to be installed and used separately. See each module's documentation for requirements and setup instructions.

[^1]: Red Rover is an [evolved](https://en.wikipedia.org/wiki/Pok%C3%A9mon_Red,_Blue,_and_Yellow) version of our previous data collection system, [rover](https://github.com/wiseLabCMU/rover), which happens to be red.

## Red Rover

<div class="grid cards" markdown>

- :material-golf-cart: [`roverc`](./roverc/index.md)

    ---

    radar + lidar + camera + imu data collection system

- :material-database-cog: [`roverd`](./roverd/index.md)

    ---

    efficient recording and storage format with an adl-compliant dataloader

- :material-factory: [`roverp`](./roverp/index.md)

    ---

    data processing and visualization tooling for red rover

- :material-database: [`i/q-1m`](./iq1m/index.md)

    ---

    one million i/q frames across indoor, outdoor, and bike-mounted settings

</div>


## See Also

<div class="grid cards" markdown>

- :material-cube-outline: [`abstract_dataloader`](https://radarml.github.io/abstract-dataloader/)

    ---

    abstract interface for composable dataloaders and preprocessing pipelines

- :material-antenna: [`xwr`](https://radarml.github.io/xwr/)

    ---

    python interface for collecting raw time signal data from TI mmWave radars

- :octicons-ai-model-16: [`nrdk`](https://radarml.github.io/nrdk/)

    ---

    neural radar development kit for deep learning on multimodal radar data

- :fontawesome-solid-hexagon-nodes: [`grt`](https://wiselabcmu.github.io/grt/)

    --- 

    *our latest work, GRT: Towards Foundational Models for Single-Chip Radar*

- :dart: [`dart`](https://wiselabcmu.github.io/dart/)

    ---

    *our prior work, DART: Implicit Doppler Tomography for Radar Novel View Synthesis*

- :material-video-wireless-outline: [`rover`](https://github.com/wiseLabCMU/rover)

    ---

    *our previous data collection platform for radar time signal*

</div>

## Development

While each red-rover module is intended to be installed and used independently, we provide a single development environment based on a shared `pyproject.toml`.

!!! warning

    While `roverc` (`red-rover/collect`) is nominally a part of the development environment (e.g., for documentation), it is not actually fully installed in the main environment. Developing `roverc` should use an actual physical data collection system, while the main environment can be used on any linux system.

**Setup**: assuming `uv` is [installed](https://docs.astral.sh/uv/getting-started/installation/), you can install all dependencies with `uv sync --all-extras` in `./red-rover`. This installs the documentation stack, testing / linting tools, and the `roverd: red-rover/format` and `roverp: red-rover/processing` modules as editable (i.e., `pip install -e`) links.

We also use [pre-commit hooks](https://pre-commit.com/); please install these with `uv run pre-commit install`.

!!! info

    You can also manually trigger the pre-commit hooks with `uv run pre-commit run`.

**Documentation**: `red-rover` has a single documentation site using `mkdocs` / `mkdocs-material` / `mkdocstrings-python`.

```sh
# Develop
uv run --extra docs mkdocs serve

# Deploy
uv run --extra docs mkdocs build
./update_gh_pages.sh
```

**Testing**: Currently, only `roverd` comes with unit tests. Run with
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
