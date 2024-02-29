"""Start data collection."""

import os
import socket
import json
import yaml


class Controller:
    """Simple socket-based CLI orchestrator."""

    def __init__(
        self, sensors: list[str] = [], addr: str = '/tmp/rover'
    ) -> None:
        self.sensors = sensors
        self.addr = addr

    @classmethod
    def from_config(cls, path: str) -> "Controller":
        """Create from `config.yaml` file."""
        with open(path) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        return cls(sensors=list(cfg.keys()))

    def _sendall(self, msg: dict) -> None:
        for sensor in self.sensors:
            try:
                tx = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                tx.connect(os.path.join(self.addr, sensor))
                tx.send(json.dumps(msg).encode())
                tx.close()
            except Exception as e:
                print(e)

    def start(self, path: str) -> None:
        self._sendall({"type": "start", "path": path})
    
    def stop(self) -> None:
        self._sendall({"type": "stop"})

    def exit(self) -> None:
        self._sendall({"type": "exit"})
