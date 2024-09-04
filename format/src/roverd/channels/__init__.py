"""Channel types."""

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
