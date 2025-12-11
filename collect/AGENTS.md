# Agent Development Guide - Data Collection (roverc)

## Overview

**roverc** is the data collection system for Red Rover, designed to collect synchronized radar, lidar, camera, and IMU data. It uses a client-server architecture where each sensor is recorded independently and asynchronously.

- **Platform**: Linux only (tested on Ubuntu 22.04)
- **Python Version**: 3.12

> **Note**: See main [AGENTS.md](../AGENTS.md) for shared code style, CLI patterns, and troubleshooting approaches.

## Module Structure

### Key Components

**Software Architecture**:
- Client-server design using Unix domain sockets
- Each sensor runs in a separate asynchronous process
- Control server coordinates all sensors and exposes Flask web GUI
- Custom Python implementation replaces Windows-only mmWave Studio

### Data Flow

1. Sensor processes collect data independently
2. Processes send logs to control server via Unix domain sockets
3. Control server manages start/stop commands
4. Flask web GUI provides interface for control and metadata
5. Data written directly to roverd format

## Development Workflow

### Setting Up Collection

**Computer Setup**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone git@github.com:RadarML/red-rover.git
cd red-rover/collect
uv sync
```

### Running Collection Scripts

Collection requires physical hardware setup and is not directly available to AI agents. The system uses CLI commands to control data collection processes.

### Validating Collected Data

Use roverd CLI tools:
```bash
roverd validate /path/to/trace
roverd info /path/to/trace
```

## Common Tasks

### Working with Collected Data

The roverc system generates data in roverd format that can be processed and analyzed:

- Use `roverd validate` to check data integrity
- Use `roverd info` to inspect trace metadata
- Check `radar/valid` channel for dropped packets
- Verify timestamp synchronization across sensors

## Best Practices

### Data Quality Checks

- Verify sensor alignment before collection
- Check lighting conditions (adjust camera gain)
- Test in representative environment first
- Monitor for dropped radar packets
- Validate timestamps across sensors

### Error Handling

- Fail-safe design: crashes only lose buffered data
- No summary metadata needed on file close
- Append-only format minimizes corruption risk
- Each sensor process is independent

### Performance Optimization

- Asynchronous sensor processes prevent blocking
- Direct binary writes (no intermediate serialization)
- Socket buffer tuning for radar capture
- Domain-specific compression (video, lidar)

## Module-Specific Dependencies

```toml
# Additional dependencies specific to roverc:
"flask"              # Control/logging web app
"pyserial"           # Serial communication
"opencv-python-headless"  # Camera interface
"ouster-sdk==0.14.0" # Lidar interface
"xwr>=0.3.3"         # TI radar interface
```

> **Style Guide**: See main [AGENTS.md](../AGENTS.md#code-style-and-conventions) for shared code style conventions.

## Troubleshooting

> **General Patterns**: See main [AGENTS.md](../AGENTS.md#troubleshooting) for shared troubleshooting approaches.

### Collection-Specific Issues

**Data Validation**:
- Check `radar/valid` channel for dropped packets
- Monitor Flask GUI logs for sensor errors
- Validate configuration YAML syntax
- Test sensors individually before full collection

## Resources

> **Shared Resources**: See main [AGENTS.md](../AGENTS.md#resources) for project links and contact info.

### Collection-Specific Documentation

- [Assembly Guide](https://radarml.github.io/red-rover/roverc/assembly/)
- [Usage Instructions](https://radarml.github.io/red-rover/roverc/usage/)
- [Data Collection Tips](https://radarml.github.io/red-rover/roverc/tips/)
