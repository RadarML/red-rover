"""Start data collection."""

import os
import socket
import json


class Controller:

    def __init__(
        self, sensors: list[str] = [], addr: str = '/tmp/rover'
    ) -> None:
        self.sensors = sensors
        self.addr = addr

    def _sendall(self, msg: dict) -> None:
        for sensor in self.sensors:
            tx = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            tx.connect(os.path.join(self.addr, sensor))
            tx.send(json.dumps(msg))
            tx.close()

    def start(self, path: str) -> None:
        self._sendall({"type": "start", "path": path})
    
    def stop(self) -> None:
        self._sendall({"type": "stop"})

    def exit(self) -> None:
        self._sendall({"type": "exit"})


def _parse(p):
    p.add_argument("-a", "--action", help="Action to run.")
    p.add_argument("-p", "--path", help="Target path (if applicable).")
    p.add_argument(
        "-s", "--sensors", nargs='+', help="Active sensors.", default=[])


def _main(args):
    ctrl = Controller(sensors=args.sensors)
    if args.action == 'start':
        ctrl.start(os.path.abspath(args.path))
    elif args.action == 'stop':
        ctrl.stop()
    elif args.action == 'exit':
        ctrl.exit()
    else:
        print("Invalid action: {}".format(args.action))
