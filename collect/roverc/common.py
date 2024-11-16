"""Common capture utilities."""

import json
import logging
import os
import socket
import threading
import traceback
from queue import Queue
from time import perf_counter, time

import numpy as np
from beartype.typing import Any, Optional
from roverd import sensors


class SensorException(Exception):
    """Sensor-related failure."""

    pass


class Capture:
    """Capture data for a generic sensor stream.

    Args:
        path: directory path to write data to.
        fps: target framerate
        report_interval: interval for reporting sensor statistics, in seconds
        log: parent logger to use
    """

    def __init__(
        self, path: str, fps: float = 1.0, report_interval: float = 5.0,
        log: Optional[logging.Logger] = None
    ) -> None:
        self.sensor = sensors.SensorData(path=path, create=True)
        self.ts: Queue = Queue()
        self.sensor.create("ts", {
            "format": "raw", "type": "f8", "shape": (),
            "description": "Timestamp, in seconds."}
        ).consume(self.ts, thread=True)

        self.len = 0
        self.period: list[float] = []
        self.runtime: list[float] = []
        self.qlen: int = 0

        self.prev_time = self.start_time = self.trace_time = perf_counter()
        self._ref_time: Optional[tuple[float, float]] = None
        self._first_loop = True

        self.fps = fps
        self.report_interval = report_interval
        self.log = log if log else logging.Logger("placeholder")

    def queue_length(self) -> int:
        """Get queue length."""
        return self.ts.qsize()

    def start(self, timestamp: Optional[float] = None) -> None:
        """Mark start of current frame processing.

        (1) Records the current time as the timestamp for this frame, and
        (2) Marks the start of time utilization calculation for this frame.
        """
        t = time() if timestamp is None else timestamp
        self.start_time = perf_counter()
        self.ts.put(np.array(t, dtype=np.float64))
        self.len += 1

    def end(self):
        """Mark end of current frame processing."""
        assert self.start_time > 0
        end = perf_counter()

        self.period.append(end - self.prev_time)
        self.runtime.append(end - self.start_time)
        self.prev_time = end
        self.qlen = max(self.queue_length(), self.qlen)

        if self.len % int(self.fps * self.report_interval) == 0:
            self.reset_stats()
        if self._first_loop:
            self._first_loop = False
            self.log.info("Receiving data.")

    def write(self, data: Any) -> None:
        """Write data."""
        raise NotImplementedError()

    def close(self) -> None:
        """Close files and clean up."""
        self.ts.put(None)

    def reset_stats(self) -> None:
        """Reset (and log) tracked statistics.

        Three values are logged:
        - `f`: frequency of the capture sensor, in Hz.
        - `u`: utilization of the capture, in percent.
        - `w`: WCET (worst-case execution time) of the capture, in ms.
        - `q`: maximum queue length.

        The log message is usually `info`, unless certain targets are violated:

        - `error`: if the observed frequency drops below 90% of the target, or
          the WCET exceeds the deadline (i.e. `T = D < C`).
        - `warning`: if the observed frequency drops below 99% of the target,
          or the WCET exceeds 90% of the deadline.
        """
        freq = 1 / np.mean(self.period)
        util = np.mean(self.runtime) * self.fps
        wcet = np.max(self.runtime)

        log_msg = (
            f"f: {freq:.2f} u: {util * 100:.0f}% "
            f"w: {wcet * 1000:.1f} q: {self.qlen}")
        if (freq < self.fps * 0.9) or (wcet * self.fps > 1.0):
            self.log.error(log_msg)
        elif (freq < self.fps * 0.99) or (wcet * self.fps > 0.9):
            self.log.warning(log_msg)
        else:
            self.log.info(log_msg)

        self.period = []
        self.runtime = []
        self.qlen = 0


class Sensor:
    """Generic sensor channel.

    Communicates using local `AF_UNIX` sockets using the socket
    `/tmp/rover/{name}` with the provided sensor name.

    NOTE: each sensor node should be initalized first.

    Args:
        name: sensor name, e.g. lidar, camera, radar
        addr: socket base address, i.e. `/tmp/rover`
        report_interval: logging interval for statistics
    """

    def __init__(
        self, name: str, addr: str = "/tmp/rover",
        report_interval: float = 10.0, fps: float = 1.0
    ) -> None:
        self.name = name
        self.log = logging.getLogger(name=name)
        self.report_interval = report_interval
        self.fps = fps

        sock = os.path.join(addr, name)
        if os.path.exists(sock):
            os.remove(sock)
        os.makedirs(addr, exist_ok=True)

        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(None)
        self._socket.bind(sock)
        self._socket.listen(1)

        self.active = False
        self._thread: Optional[threading.Thread] = None
        self.frame_count = 0

    def capture(self, path: str) -> None:
        """Run capture; should check `self.active` on every loop."""
        raise NotImplementedError()

    def close(self) -> None:
        """Clean up sensor."""
        pass

    def _start_capture(self, path: Optional[str]) -> None:
        """Start capture."""
        if self.active:
            self.log.error("Tried to start capture when another is active.")
        elif path is None:
            self.log.error("A valid path must be provided to start.")
        else:
            def _capture_wrapped():
                self.frame_count = 0
                try:
                    self.capture(path)
                except Exception as e:
                    self.log.critical(repr(e))
                    self.log.debug(''.join(traceback.format_exception(e)))
                finally:
                    self.active = False

            self.log.info("Starting capture: {}".format(path))
            self.active = True
            self._thread = threading.Thread(target=_capture_wrapped)
            self._thread.start()

    def _stop_capture(self) -> None:
        """End capture."""
        if not self.active:
            return

        self.log.debug("Stopping capture...")
        self.active = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.log.info("Stopped capture.")

    def _recv(self) -> dict:
        """Receive message (spins until a valid message is received)."""
        while True:
            connection, _ = self._socket.accept()
            data = connection.recv(4096).decode()
            connection.close()

            try:
                return json.loads(data)
            except json.JSONDecodeError:
                self.log.error("Invalid json: {}".format(data))

    def loop(self) -> None:
        """Main control loop (spins until `exit` is received)."""
        while True:
            cmd = self._recv()
            cmd_type = cmd.get("type")
            if cmd_type == 'start':
                self._start_capture(cmd.get("path", ""))
            elif cmd_type == 'stop':
                self._stop_capture()
            elif cmd_type == 'exit':
                self.log.info("Exiting...")
                self._stop_capture()
                break
            else:
                self.log.error("Invalid command type: {}".format(cmd_type))
        self.close()
