"""Extract a subset of a single trace."""

import os

import numpy as np

from roverd import Trace
from roverd.sensors import DynamicSensor


def cli_extract(
    src: str, dst: str, /, start: float | None = None, end: float | None = None,
    length: float | None = None, relative: bool = False
) -> None:
    """Extract a subset of a trace.

    !!! info

        Two of `start`, `end`, and `length` must be specified. If `relative`,
        these values are specified as a proportion of the trace duration.

    Args:
        src: path to the trace directory.
        dst: output trace directory.
        start: start time offset relative to the trace start.
        end: end time offset relative to the trace start (if positive) or
            trace end (if negative).
        length: length of the extracted trace in seconds.
        relative: whether the start/end/length values are relative to the trace
            duration, in seconds.
    """
    trace = Trace.from_config(src)

    trace_start = max(v.metadata.timestamps[0] for v in trace.sensors.values())
    trace_end = min(v.metadata.timestamps[-1] for v in trace.sensors.values())
    duration = trace_end - trace_start

    if sum(x is not None for x in (start, end, length)) < 2:
        raise ValueError(
            "Two of `start`, `end`, and `length` must be specified.")

    if end is None and start is not None and length is not None:
        end = start + length
    if start is None and end is not None and length is not None:
        start = end - length

    if relative:
        start = trace_start + (start * duration if start is not None else 0)
        end = trace_start + (end * duration if end is not None else 0)

    if os.path.exists(dst):
        raise FileExistsError(f"Output directory {dst} already exists.")

    os.makedirs(dst)
    for s_name, sensor in trace.sensors.items():
        assert isinstance(sensor, DynamicSensor)
        s_copy = DynamicSensor(os.path.join(dst, s_name), create=True)

        i_start, i_end = np.searchsorted(
            sensor.metadata.timestamps, np.array([start, end]))

        for ch_name, channel in sensor.channels.items():
            ch_copy = s_copy.create(ch_name, sensor.config[ch_name])
            ch_copy.write(channel.read(i_start, samples=i_end - i_start))
