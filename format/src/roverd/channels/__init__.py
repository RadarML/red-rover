"""Channel types.

Channels are conventionally specified by a configuration dict with the
following fields:

- `format`: channel data format. Currently supports:

    - `raw` (:py:class:`.RawChannel`): little-endian raw byte array
    - `lzma` (:py:class:`.LzmaChannel`): lzma-compressed `raw` data
    - `lzmaf` (:py:class:`.LzmaFrameChannel`): lzma-compressed, but each
      "frame" (data at a time step) is compressed independently to allow for
      random reading.
    - `mjpg` (:py:class:`.VideoChannel`): mjpeg video data.

- `type`: data type, using numpy size-in-bytes convention (e.g. u2 for
  2-byte/16-bit unsigned integer)
- `shape`: shape of the non-time dimensions.
- `desc`: description of the channel; should be human-readable, and is not
  intended for use by scripts.

For example::

    {
        "format": "lzmaf", "type": "u1", "shape": [64, 128],
        "desc": "range-azimuth BEV from 3D polar occupancy"
    }
"""

from .base import Channel, Prefetch
from .lzma import LzmaChannel, LzmaFrameChannel
from .raw import RawChannel
from .video import VideoChannel

CHANNEL_TYPES: dict[str, type[Channel]] = {
    "raw": RawChannel,
    "lzma": LzmaChannel,
    "lzmaf": LzmaFrameChannel,
    "mjpg": VideoChannel
}

__all__ = [
    "Prefetch", "Channel",
    "RawChannel", "LzmaChannel", "LzmaFrameChannel", "VideoChannel"
]
