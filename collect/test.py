from radar_api import AWRSystem
import logging


logging.basicConfig(level=logging.INFO)
radar = AWRSystem()
i = 0
for data in radar.stream():
    i += 1
    print(data.data.shape)
    if i > 10:
        break

radar.stop()
