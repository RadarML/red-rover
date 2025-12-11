# Agent Development Guide - Data Processing (roverp)

## Overview

**roverp** provides data processing and visualization tooling for Red Rover datasets. It includes CLI tools for anonymization, semantic segmentation, pose interpolation.

- **Platform**: Cross-platform (Linux, Windows, macOS)
- **Python Version**: 3.12, 3.13

> **Note**: See main [AGENTS.md](../AGENTS.md) for shared installation, code style, and troubleshooting patterns.

## Module Structure

### Key Components

**CLI Tools**:
- `roverp anonymize` - Blur faces in camera data
- `roverp sensorpose` - Interpolate poses for specific sensors
- `roverp report` - Generate speed reports
- `roverp segment` - Run semantic segmentation on video

**Processing Components**:
- Pose interpolation and alignment
- Semantic segmentation (SegFormer-B5 fine-tuned on ADE20K)
- Face anonymization (RetinaFace)
- Point cloud generation
- Trajectory visualization

### Processing Pipeline

**Typical Workflow**:
1. **Collect data** (roverc) → roverd format
2. **Pose interpolation** → `roverp sensorpose`
3. **Optional**: Anonymize video → `roverp anonymize`
4. **Optional**: Semantic segmentation → `roverp segment`
5. **Analysis/Training** → Use with ML frameworks

## Development Workflow

> **Installation**: See main [AGENTS.md](../AGENTS.md#setting-up-the-environment) for installation patterns.

⚠️ **Important**: Don't use `--all-extras` - processing extras are heavy!

**Module-Specific Extras**:
- `anonymize`: Face detection/blurring
- `pcd`: Point cloud format
- `ply`: PLY format
- `semseg`: Semantic segmentation (torch, transformers)

> **Code Style**: See main [AGENTS.md](../AGENTS.md#code-style-and-conventions) for shared style guide.

### Debug Strategies

> **General CLI**: See main [AGENTS.md](../AGENTS.md#cli-usage) for common CLI debugging patterns.

**Processing-Specific Debugging**:
```python
from roverp.readers import load_trajectory, interpolate_poses

# Test trajectory loading
trajectory = load_trajectory("trajectory.csv")
print(trajectory.shape)

# Validate interpolation
poses = interpolate_poses(trajectory, timestamps)
assert len(poses['position']) == len(timestamps)
```

## Resources

> **Shared Resources**: See main [AGENTS.md](../AGENTS.md#resources) for project links and contact info.

### Processing-Specific Documentation

- [CLI Reference](https://radarml.github.io/red-rover/roverp/cli/)
- [Graphics Tools](https://radarml.github.io/red-rover/roverp/graphics/)

### Algorithm Documentation

- [SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)
- [RetinaFace](https://github.com/serengil/retinaface)
- [ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
