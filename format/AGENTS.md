# Agent Development Guide - Data Formatting (roverd)

## Overview

**roverd** is the data interface and storage format for Red Rover. It provides an extremely simple, flexible, and extendable binary data format with an [abstract-dataloader](https://radarml.github.io/abstract-dataloader/)-compliant API.

- **Platform**: Cross-platform (Linux, Windows, macOS)
- **Python Version**: 3.10, 3.11, 3.12, 3.13

> **Note**: See main [AGENTS.md](../AGENTS.md) for shared installation, code style, and troubleshooting.

**TL;DR**: Simple binary containers + native filesystem + JSON/YAML metadata = easy ML training and data collection.

## Module Structure

### Key Components

**Data Format**:
- Channel-based organization (one channel per sensor stream)
- Binary files for time-series data
- JSON/YAML metadata files (human-readable)
- Native filesystem (no filesystem-within-a-filesystem)
- Domain-specific compression (video, lzma for lidar)

**Python API**:
- Abstract-dataloader-compliant
- Random access for ML training
- Lazy loading and caching
- Composable transforms pipeline
- Type-safe with beartype and jaxtyping

**CLI Tools** (via `roverd` command):
- `roverd info` - Display trace information
- `roverd validate` - Validate trace integrity
- `roverd list` - List channels in trace
- `roverd extract` - Extract channel data
- `roverd blobify` - Convert to blob format
- `roverd checksum` - Compute checksums

### Data Formats

**Supported Channel Types**:
- Raw binary arrays (`.npz` format)
- Compressed binary (lzma with index files)
- Video (`.avi`, MJPEG or other codecs)
- Timestamps (raw binary)
- Metadata (JSON/YAML)

**Typical Trace Structure**:
```
sequence/
├── radar/
│   ├── meta.json          # Channel metadata
│   ├── radar.json         # Radar intrinsics
│   ├── iq                 # Raw complex time signal
│   ├── ts                 # Timestamps
│   └── valid              # Frame validity flags
├── lidar/
│   ├── meta.json
│   ├── lidar.json
│   ├── rng                # Compressed range data
│   ├── rng_i              # Index for random access
│   ├── nir                # Near-infrared
│   ├── nir_i
│   └── ts
├── camera/
│   ├── meta.json
│   ├── video.avi          # MJPEG video
│   └── ts
├── imu/
│   ├── meta.json
│   ├── acc                # Acceleration
│   ├── avel               # Angular velocity
│   ├── rot                # Rotation
│   └── ts
└── config.yaml            # Collection configuration
```

## Development Workflow

### Setting Up the Environment

> **Installation**: See main [AGENTS.md](../AGENTS.md#setting-up-the-environment) for installation patterns.

**Module-Specific Extras**:
- `video`: OpenCV for camera data
- `ouster`: Ouster SDK for lidar transforms

### Running Format Scripts

**Validate a trace**:
```bash
roverd validate /path/to/trace
```

**Get trace info**:
```bash
roverd info /path/to/trace
```

**List channels**:
```bash
roverd list /path/to/trace
```

**Extract channel data**:
```bash
roverd extract /path/to/trace --channel radar/iq --output data.npy
```

### Validating Output

> **CLI Patterns**: See main [AGENTS.md](../AGENTS.md#cli-usage) for common CLI usage patterns.
> **Testing**: See main [AGENTS.md](../AGENTS.md#testing-patterns) for test setup and requirements.

## Module-Specific Dependencies

```toml
# Additional dependencies specific to roverd:
"optree >= 0.16.0"           # Tree operations
"abstract_dataloader >= 0.3.5" # Data loading interface
"einops>=0.7.0"              # Array operations
```

> **Code Style**: See main [AGENTS.md](../AGENTS.md#code-style-and-conventions) for shared style guide.
> **Testing**: See main [AGENTS.md](../AGENTS.md#testing-patterns) for testing requirements.

## Troubleshooting

### Common Format Issues

**Invalid trace structure**:
```bash
roverd validate /path/to/trace
# Check for missing meta.json, mismatched shapes
```

**Timestamp issues**:
- Verify monotonicity
- Check for duplicate timestamps
- Validate timestamp alignment across sensors

**Compression errors**:
- Check index file (`_i` suffix) exists
- Verify index matches compressed data length
- Test decompression of random frames

**Memory mapping failures**:
- Ensure file size matches metadata shape
- Check file permissions
- Verify dtype matches metadata

### Debug Strategies

> **General Debugging**: See main [AGENTS.md](../AGENTS.md#shared-patterns) for common validation patterns.

**Format-Specific Debugging**:
```python
from roverd import Trace

trace = Trace("/path/to/trace")
print(trace.channels)        # List all channels
print(trace.radar.iq.shape)  # Check shape
print(trace.radar.ts[:10])   # Check timestamps
```

## Resources

> **Shared Resources**: See main [AGENTS.md](../AGENTS.md#resources) for project links and contact info.

### Format-Specific Documentation

- [Roverd Documentation](https://radarml.github.io/red-rover/roverd/)
- [Channel Specifications](https://radarml.github.io/red-rover/roverd/channels/)
- [API Reference](https://radarml.github.io/red-rover/roverd/api/)
- [Transform Pipeline](https://radarml.github.io/red-rover/roverd/transforms/)
- [ouster-sdk](https://static.ouster.dev/sdk-docs/) - Lidar processing
- [OpenCV](https://opencv.org/) - Video I/O
