"""Rover Control Server.

Run with
```sh
ROVER_CFG=/path/to/config flask run --host=0.0.0.0
```
"""

import io
import json
import logging
import os
import subprocess
import threading
from datetime import datetime

import yaml
from controller import Controller
from flask import Flask, jsonify, render_template, request


class RoverSensor:
    """Data collection process wrapper.

    Args:
        sensor: name of the sensor to collect data from (radar, lidar, etc).
    """

    def __init__(self, sensor: str) -> None:
        self.name = sensor
        self.log: list[tuple[float, dict]] = []
        self.log_tail = 0

        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        """Run / log loop."""
        with subprocess.Popen(
            ['./.venv/bin/python', 'cli.py', "run", "--sensor", self.name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as proc:
            for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):  # type: ignore
                print(line.rstrip())
                try:
                    data = json.loads(line.rstrip())
                    ts = datetime.strptime(data['ts'], "%Y-%m-%d %H:%M:%S,%f")
                    self.log.append((ts.timestamp(), data))
                    if len(self.log) > 1024:
                        self.log = self.log[128:]
                        self.log_tail += 128

                except json.JSONDecodeError:
                    print("Invalid json: {}".format(line.rstrip()))

    def log_entries(self, start: float) -> list[tuple[float, dict]]:
        """Aggregate log entries more recent than the start time.

        Args:
            start: timestamp in seconds since epoch; if `<=0`, return all.

        Returns:
            A list of tuples with timestamp and log entry, where the timestamp
                is in seconds since epoch.
        """
        return [(ts, entry) for ts, entry in self.log if ts > start]


class Rover:
    """Rover system."""

    def __init__(self):
        with open(os.getenv('ROVER_CFG')) as f:  # type: ignore
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.controller = Controller(list(self.cfg.keys()))
        self.sensors = {s: RoverSensor(s) for s in self.cfg}

    def log(self, start: float) -> dict:
        """Get log entries.

        Args:
            start: timestamp in seconds since epoch; if `<=0`, return all.

        Returns:
            A dictionary with the log entries for each sensor
                (`entries/<sensor_name>:dict`), and the timestamp of the
                last entry (`ts:str`) in `%Y-%m-%dT%H:%M:%S,%f` format.
        """
        if start < 0:
            msgs = {k: v.log for k, v in self.sensors.items()}
        else:
            msgs = {k: v.log_entries(start) for k, v in self.sensors.items()}

        ts_last = 0
        for v in self.sensors.values():
            if len(v.log) > 0:
                ts_last = max(ts_last, v.log[-1][0])
        return {
            "entries": {k: [entry for _, entry in v] for k, v in msgs.items()},
            "ts": datetime.strftime(
                datetime.fromtimestamp(ts_last), "%Y-%m-%dT%H:%M:%S,%f")}


def _get_version_info():
    output = subprocess.check_output(
        "git log -1 --format='%h %at'", shell=True, text=True)
    commit, raw = output.strip().split()
    ts = datetime.fromtimestamp(int(raw)).strftime("%m/%d/%Y %H:%M")
    return f"{commit} @ {ts}"


app = Flask(__name__)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

with app.app_context():
    version = _get_version_info()
    rover = Rover()


@app.route('/')
def index():
    """Index page.

    App route: `GET:/`

    Response: Rendered HTML.
    """
    media = "/media/rover"
    disks = os.listdir(media)
    return render_template(
        "index.html", version=version, media=media, disks=disks)


@app.route('/command', methods=['POST'])
def command():
    """Issue command.

    App route: `POST:/command`

    Response:
        - `200`: "ok" if command was accepted
        - `400`: an error message with if the command is invalid or
            arguments are missing.
    """
    try:
        action = request.json['action']  # type: ignore
        if action == 'start':
            path = request.json["path"]  # type: ignore
            if path.endswith('/'):
                path += datetime.now().strftime("%Y-%m-%d.%H-%M-%S")

            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, 'config.yaml'), 'w') as f:
                yaml.dump(rover.cfg, f)

            rover.controller.start(path)
            return "ok"
        elif action == 'stop':
            rover.controller.stop()
            return "ok"
        else:
            return "invalid action", 400
    except KeyError:
        return "missing arguments", 400


@app.route('/log')
def log_all():
    """Get all log messages.

    App route: `GET:/log`

    Response: JSON object with log entries for each sensor and the timestamp of
    the last entry. The log entries are in the format
    ```
    {
        "entries": {
            "<sensor_name>": [
                {
                    "lvl": <log_level>,
                    "msg": "<log_message>",
                    "mod": "<module_name>"
                },
                ...
            ],
            ...
        },
        "ts": "<timestamp>"
    }
    ```
    """
    return jsonify(rover.log(start=-1.))


@app.route('/log/<start>')
def log(start=None):
    """Get log messages after start time.

    App route: `GET:/log/<start>`

    Response: JSON object with log entries for each sensor and the timestamp of
    the last entry.
    """
    try:
        ts_start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S,%f")  # type: ignore
        return jsonify(rover.log(start=ts_start.timestamp()))
    except (TypeError, ValueError):
        return "bad timestamp", 400


if __name__ == '__main__':
    app.run(debug=False)
