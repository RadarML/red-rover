"""Run cartographer, and kill (send ctrl+C) on completion.

Usage: `python kill_cartographer.py executable args ...`
"""

import subprocess
import signal
import sys
import re
import os


done_pattern = r"\[cartographer_offline_node-(.*)\] process has finished cleanly(.*)"
process = subprocess.Popen(
    sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    env=os.environ)

while True:
    line = process.stdout.readline()
    if not line:
        print("done?")
        break
    print(line.decode())

    # if re.match(done_pattern, line):
    #     process.send_signal(signal.SIGINT)


