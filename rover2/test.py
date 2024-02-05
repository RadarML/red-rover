import sys
import time
import logging
from radar import DCA1000EVM, RadarDataWriter
from radar.dca_types import LVDS


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
dca = DCA1000EVM()
dca.setup(delay=25, lvds=LVDS.TWO_LANE)
dca.start(RadarDataWriter("tmp"))
# time.sleep(10)
# dca.stop()
