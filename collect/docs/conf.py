import os, sys

os.chdir("..")
sys.path.insert(0, os.getenv("ROOT_CONF_DIR"))
sys.path.insert(0, os.path.abspath('.'))

project = 'Rover - Collect'

from conf import *
