# Agent Development Guide

## Overview

This guide provides instructions for AI coding agents working with the **Red Rover** radar dataset ecosystem - an end-to-end system for collecting, loading, and processing mmWave radar time signal data along with lidar and camera data.

Red Rover is packaged as a mono-repo with three main modules that can be installed and used independently:

- **roverc** (`collect/`) - Radar + lidar + camera + IMU data collection system
- **roverd** (`format/`) - Efficient recording/storage format with abstract-dataloader-compliant API
- **roverp** (`processing/`) - Data processing and visualization tooling

## Repository Structure

### Key Directories

- `collect/` - Data collection system (roverc) - Linux-only, Python 3.12
- `format/` - Data format and API (roverd) - Cross-platform, Python 3.10+
- `iq1m/` - I/Q-1M dataset (1M radar-lidar-camera samples)
- `processing/` - Processing utilities (roverp) - Python 3.12+
- `docs/` - MkDocs documentation source files
- `site/` - Generated documentation site (auto-deployed on push to main)

### Configuration Files

- `pyproject.toml` - Root project dependencies and metadata (version 0.3.3)
- `mkdocs.yml` - Documentation configuration
- Each submodule has its own `pyproject.toml`

## Development Workflow

### Setting Up the Environment

**Prerequisites**: Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

**Full Development Setup**:
```bash
cd red-rover
uv sync --all-extras  # Installs all modules + docs + dev tools
uv run pre-commit install  # Install pre-commit hooks
```

**Note**: `roverc` is designed for physical data collection systems and is not fully installed in the main dev environment.

### Running Tests

Currently only `roverd` has unit tests:
```bash
export ROVERD_TEST_TRACE=/data/roverd/test-trace  # Point to test data
uv run --all-extras pytest -ra --cov --cov-report=html --cov-report=term
```

View coverage report:
```bash
uv run python -m http.server 8001 -d ./htmlcov
```

### Run Type Checker

While the type checker runs in the CI and pre-commit hooks, and will run automatically if you configure the appropriate IDE extensions, you can manually invoke it with:
```bash
uv run pyright
```

### Building Documentation

Serve documentation locally:
```bash
uv run --extra docs mkdocs serve
```

Documentation is automatically built and deployed by GitHub Actions on push to `main`.

## Common Tasks

### Version Management

Red Rover uses a mono-repo versioning strategy where all modules share the same version number.

**When updating versions**:
1. Update version in each submodule's `pyproject.toml` (`collect/`, `format/`, `processing/`)
2. Update version in root `pyproject.toml`
3. **CRITICAL**: Regenerate `uv.lock`:
   ```bash
   uv sync --all-extras --upgrade
   ```

### Adding New Features

- Follow the code style guidelines below
- Add type annotations (project uses beartype and jaxtyping)
- Update documentation in `docs/` directory
- Run pre-commit hooks before committing

### Updating Documentation

- Documentation uses MkDocs with Material theme and mkdocstrings-python
- Edit markdown files in `docs/` directory
- Use `:::module.path` syntax to auto-generate API docs
- Test locally with `uv run mkdocs serve`

### Processing Data

See module-specific AGENTS.md files:
- `collect/AGENTS.md` - Data collection workflows
- `format/AGENTS.md` - Data format and loading
- `processing/AGENTS.md` - Processing pipelines

## Code Style and Conventions

### Python Style Guide

**Ruff Configuration** (applies to root and processing):
- Line length: 80 characters
- Indent: 4 spaces
- Selected lints: W (warnings), N (naming), I (imports), D (docstrings), NPY (numpy)

**Type Checking**:
- All code is fully typed
- Uses beartype for runtime type checking
- Uses jaxtyping for array shape/dtype checking

**Common Dependencies**:
- `numpy` - Array operations
- `beartype` + `jaxtyping` - Type checking
- `pyyaml` - Configuration files
- `tyro` - CLI argument parsing
- `abstract_dataloader` - Data loading interface

**Naming Conventions**:
- Mathematical variables can use capitalization (N803, N806 ignored)
- Exception names don't need "Error" suffix (N818 ignored)

### Documentation Standards

**Docstring Style**:
- Use Google-style docstrings
- Multi-line summary on first line (D213 preferred)
- No blank line before class (D203 preferred)
- Inheriting docstrings is allowed (D102 ignored)
- Class initialization goes in class docstring (D107 ignored)
- No blank lines after section headers (D413 - preferred by mkdocstrings)

**Documentation Format**:
- Use MkDocs markdown with Material theme extensions
- Use admonitions: `!!! info`, `!!! warning`, `!!! tip`, `!!! danger`
- Code blocks with language tags
- Use grid cards for navigation

## Shared Patterns

### CLI Usage

**Common CLI Patterns** (using [tyro](https://brentyi.github.io/tyro/)):
- Positional arguments: command line positionals
- Named arguments: flagged arguments (e.g., `--channel`, `--output`)
- Help: `{command} --help`
- Verbose: `{command} --verbose` (where available)

### Data Validation

**Standard Validation**:
```bash
roverd validate /path/to/trace    # Check trace integrity
roverd info /path/to/trace        # Display trace information
roverd list /path/to/trace        # List available channels
```

**Common Validation Checks**:
- Verify timestamp monotonicity across sensors
- Check file sizes match metadata shapes
- Validate channel data can be read
- Test random access (first, middle, last frames)

### Testing Patterns

**Test Environment Setup**:
```bash
export ROVERD_TEST_TRACE=/data/roverd/test-trace  # Point to test data
pytest -ra --cov --cov-report=html --cov-report=term
```

**Coverage Requirements**:
- Test trace needs ≥0.5s data across all modalities
- Test random access patterns
- Validate all supported channel types
- Check transform pipeline functionality

## Shared Patterns

### CLI Usage

**Common CLI Patterns** (using [tyro](https://brentyi.github.io/tyro/)):
- Positional arguments: command line positionals
- Named arguments: flagged arguments (e.g., `--channel`, `--output`)
- Help: `{command} --help`
- Verbose: `{command} --verbose` (where available)

### Data Validation

**Standard Validation**:
```bash
roverd validate /path/to/trace    # Check trace integrity
roverd info /path/to/trace        # Display trace information
roverd list /path/to/trace        # List available channels
```

**Common Validation Checks**:
- Verify timestamp monotonicity across sensors
- Check file sizes match metadata shapes
- Validate channel data can be read
- Test random access (first, middle, last frames)

### Testing Patterns

**Test Environment Setup**:
```bash
export ROVERD_TEST_TRACE=/data/roverd/test-trace  # Point to test data
pytest -ra --cov --cov-report=html --cov-report=term
```

**Coverage Requirements**:
- Test trace needs ≥0.5s data across all modalities
- Test random access patterns
- Validate all supported channel types
- Check transform pipeline functionality

## Troubleshooting

### Common Issues

**Mono-repo Dependencies**:
- If you change versions in submodules, always regenerate `uv.lock`
- Use `uv sync --all-extras --upgrade` after version bumps

**Pre-commit Hooks**:
- Manually trigger with: `uv run pre-commit run`
- Re-install after updates: `uv run pre-commit install`

**Documentation Build**:
- Ensure all `--extra docs` dependencies are installed
- Check for broken cross-references in API docs

### Debug Strategies

- Use coverage reports to identify untested code paths
- Check GitHub Actions logs for CI/deployment issues
- Verify type annotations with beartype runtime checks

## Related Projects

- [abstract_dataloader](https://radarml.github.io/abstract-dataloader/) - Abstract interface for composable dataloaders
- [xwr](https://radarml.github.io/xwr/) - Python interface for TI mmWave radars
- [nrdk](https://radarml.github.io/nrdk/) - Neural radar development kit
- [GRT](https://wiselabcmu.github.io/grt/) - Foundational models for single-chip radar
