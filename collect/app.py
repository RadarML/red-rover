"""Rover Control Server."""

import os
import yaml, json
import subprocess
import threading
import io
import logging
from datetime import datetime

from roverc_scripts import Controller
from flask import Flask, render_template, jsonify, request


class RoverSensor:
    """Data collection process wrapper."""

    def __init__(self, sensor: str) -> None:
        self.name = sensor
        self.log: list[tuple[float, dict]] = []
        self.log_tail = 0

        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def loop(self):
        """Run / log loop."""
        with subprocess.Popen(
            ['./env/bin/python', 'collect.py', "run", "-s", self.name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as proc:
            for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
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

    def log_entries(self, start: float):
        """Aggregate log entries more recent than the start time."""
        return [(ts, entry) for ts, entry in self.log if ts > start]


class Rover:
    """Rover system."""

    def __init__(self):
        with open(os.getenv('ROVER_CFG')) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.controller = Controller(list(self.cfg.keys()))
        self.sensors = {s: RoverSensor(s) for s in self.cfg}

    def log(self, start: float):
        """Get log entries."""
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
    media = "/media/rover"
    disks = os.listdir(media)
    return render_template(
        "index.html", version=version, media=media, disks=disks)


@app.route('/command', methods=['POST'])
def command():
    try:
        action = request.json['action']
        if action == 'start':
            path = request.json["path"]
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
    return jsonify(rover.log(start=-1.))


@app.route('/log/<start>')
def log(start=None):
    try:
        ts_start = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S,%f")
        return jsonify(rover.log(start=ts_start.timestamp()))
    except (TypeError, ValueError):
        return "bad timestamp", 400


if __name__ == '__main__':
    app.run(debug=False)
