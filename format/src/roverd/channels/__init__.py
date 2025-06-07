"""Channel types.

Channels are conventionally specified by a configuration dict with the
following fields:

- `format`: channel data format.
- `type`: data type, using numpy size-in-bytes convention (e.g. u2 for
  2-byte/16-bit unsigned integer)
- `shape`: shape of the non-time dimensions.
- `desc`: description of the channel; should be human-readable, and is not
  intended for use by scripts.

???+ quote "Sample Configuration"

    ```json
    {
        "format": "lzmaf", "type": "u1", "shape": [64, 128],
        "desc": "range-azimuth BEV from 3D polar occupancy"
    }
    ```

Currently supported channel types:

| Name    | Class                   | Description                            |
| ------- | ----------------------- | -------------------------------------- |
| `raw`   | [`RawChannel`][.]       | Little-endian raw byte array           |
| `lzma`  | [`LzmaChannel`][.]      | LZMA-compressed raw data               |
| `lzmaf` | [`LzmaFrameChannel`][.] | LZMA, but each frame is compressed independently |
| `mjpg`  | [`VideoChannel`][.]     | MJPEG video                            |
"""

from typing import Sequence

from . import utils
from .base import Channel
from .lzma import LzmaChannel, LzmaFrameChannel
from .raw import RawChannel
from .video import VideoChannel

CHANNEL_TYPES: dict[str, type[Channel]] = {
    "raw": RawChannel,
    "lzma": LzmaChannel,
    "lzmaf": LzmaFrameChannel,
    "mjpg": VideoChannel
}


def from_config(
    path: str, format: str, type: str, shape: Sequence[int],
    description: str | None = None, desc: str | None = None
) -> Channel:
    """Create channel from configuration.

    Args:
        path: File path to the channel data.
        format: channel format name.
        type: data type, using numpy size-in-bytes convention (e.g. u2 for
            2-byte/16-bit unsigned integer).
        shape: shape of the non-time dimensions.

    Returns:
        Initialized channel object.
    """
    ctype = CHANNEL_TYPES.get(format)
    if ctype is None:
        raise ValueError(f"Unknown channel format: {format}")
    return ctype(path=path, dtype=type, shape=shape)


__all__ = [
    "utils", "Channel",
    "RawChannel", "LzmaChannel", "LzmaFrameChannel", "VideoChannel"
]
