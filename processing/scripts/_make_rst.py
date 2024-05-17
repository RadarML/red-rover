"""Make .rst documentation generator."""

import os

print("Scripts")
print("=======")


for script in sorted(os.listdir(".")):
    script = script.split('.')[0]
    if not script.startswith("_"):
        print("""
{script}
{sep}

.. argparse::
   :func: _parser
   :filename: ../process.py
   :prog: process.py
   :path: {script}""".format(script=script, sep='-' * len(script)))
