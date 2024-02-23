"""Common capture utilities."""

import os, sys
import json
import yaml
import struct
import socket
import logging
import threading
import traceback
from time import perf_counter, time

import numpy as np

from beartype.typing import Callable, Optional, Any


SensorMetadata = dict[str, dict[str, Any]]


class SensorException(Exception):
    """Sensor-related failure."""
    pass


class BaseCapture:
    """Capture data for a generic sensor stream.
    
    Parameters
    ----------
    path: directory path to write data to.
    fps: target framerate
    report_interval: interval for reporting sensor statistics, in seconds
    log: parent logger to use
    kwargs: passthrough to sensor-specific initialization    
    """

    _STATS: dict[str, Callable[[np.ndarray], float]] = {
        "mean": np.mean,
        "p1": lambda x: np.percentile(x, 1),
        "p99": lambda x: np.percentile(x, 99)
    }

    _COMMON: SensorMetadata = {
        "ts": {
            "format": "raw", "type": "f64", "shape": (),
            "description": "Timestamp, in seconds."},
        "util": {
            "format": "raw", "type": "f32", "shape": (),
            "description": "System utilization fraction for data processing."}
    }

    def _init(self, path: str, **kwargs) -> SensorMetadata:
        """Run sensor-specific initialization.

        Returns
        -------
        Data channel metadata information.
        """
        raise NotImplementedError()

    def __init__(
        self, path: str, fps: float = 1.0, report_interval: float = 10.0,
        log: Optional[logging.Logger] = None, **kwargs
    ) -> None:
        os.makedirs(path)
        self.ts = open(os.path.join(path, "ts"), 'wb')
        self.util = open(os.path.join(path, "util"), 'wb')

        self.len = 0
        self.period: list[float] = []
        self.runtime: list[float] = []
        self.prev_time = self.start_time = self.trace_time = perf_counter()
        self._ref_time: Optional[tuple[float, float]] = None

        self.fps = fps
        self.report_interval = report_interval
        self.log = log if log else logging.Logger("placeholder")

        meta = self._init(path, **kwargs)
        with open(os.path.join(path, "meta.json"), 'w') as f:
            json.dump({**meta, **self._COMMON}, f, indent=4)


    def start(self, timestamp: Optional[float] = None) -> None:
        """Mark start of current frame processing.

        (1) Records the current time as the timestamp for this frame, and
        (2) Marks the start of time utilization calculation for this frame.

        Parameters
        ----------
        timestamp: if not `None`, use an external timestamp (i.e. from a low
            level sensor API) instead of the current system time. This
            timestamp is converted into the system time using the first time
            `.start()` is called with a timestamp.
        """
        if timestamp is None:
            t = time()
        else:
            if self._ref_time is None:
                self._ref_time = (timestamp, time())
            t = (timestamp - self._ref_time[0]) + self._ref_time[1]

        self.start_time = perf_counter()
        self.ts.write(struct.pack('d', t))
        self.len += 1

    def end(self):
        """Mark end of current frame processing."""
        assert self.start_time > 0
        end = perf_counter()

        self.period.append(end - self.prev_time)
        self.runtime.append(end - self.start_time)
        self.prev_time = end
        self.util.write(struct.pack('f', end - self.start_time))

        if self.len % int(self.fps * self.report_interval) == 0:
            self.reset_stats()

    def write(self, *args, **kwargs) -> None:
        """Write a single frame."""
        raise NotImplementedError()

    def close(self) -> None:
        """Close files and clean up."""
        self.ts.close()
        self.util.close()

    def reset_stats(self) -> None:
        """Reset tracked statistics."""
        period = np.array(self.period)
        runtime = np.array(self.runtime)
        self.log.info("freq: {}  util: {}".format(
            " ".join("{}={:5.2f}".format(
                k, 1 / v(period)) for k, v in self._STATS.items()),
            " ".join("{}={:5.2f}".format(
                k, self.fps * v(runtime)) for k, v in self._STATS.items())))
        self.period = []
        self.runtime = []


class BaseSensor:
    """Generic sensor channel.
    
    Communicates using local `AF_UNIX` sockets using the socket
    `/tmp/rover/{name}` with the provided sensor name.

    NOTE: each sensor node should be initalized first.

    Parameters
    ----------
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

    @classmethod
    def from_config(cls, name: str, path: str) -> "BaseSensor":
        """Initalize from a entry in a config file."""
        with open(path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)[name]["args"]
        return cls(name=name, **cfg)

    def capture(self, path: str) -> None:
        """Run capture; should check `self.active` on every loop."""
        raise NotImplementedError()

    def close(self) -> None:
        """Clean up sensor."""
        pass

    def _start_capture(self, path: Optional[str]) -> None:
        """Start capture."""
        def _capture_wrapped():
            self.frame_count = 0
            try:
                self.capture(path)
            except Exception as e:
                self.log.critical(repr(e))
                self.log.debug(''.join(traceback.format_exception(e)))
            finally:
                self.active = False

        if self.active:
            self.log.error("Tried to start capture when another is active.")
        elif path is None:
            self.log.error("A valid path must be provided to start.")
        else:
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

    @classmethod
    def main(cls) -> None:
        """Run as a independent script."""
        try:
            logging.basicConfig(level=logging.DEBUG)
            cls.from_config(*sys.argv[1:]).loop()
            exit(0)
        except SensorException:
            exit(-1)
