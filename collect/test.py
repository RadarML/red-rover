import logging
from radar_api import AWR1843, DCA1000EVM, RadarDataWriter


logging.basicConfig(level=logging.INFO)

awr = AWR1843()
awr.setup()
try:
    awr.start()
except Exception as e:
    print(e)

dca = DCA1000EVM()
dca.setup()
writer = RadarDataWriter("test", chirp_len=256)
dca.start(writer)

