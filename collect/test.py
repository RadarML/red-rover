import logging
import time
from radar_api import AWR1843, DCA1000EVM, RadarDataWriter


logging.basicConfig(level=logging.INFO)

awr = AWR1843()
# awr.setup_from_config("test.cfg")
awr.setup()
awr.start()

dca = DCA1000EVM()
dca.setup()
writer = RadarDataWriter("test", chirp_len=256)
dca.start(writer)
time.sleep(5)
dca.stop()
