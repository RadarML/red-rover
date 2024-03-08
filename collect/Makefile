#     ______  _____  _    _ _______  ______
#    |_____/ |     |  \  /  |______ |_____/
#    |    \_ |_____|   \/   |______ |    \_
#    Radar sensor fusion research platform
#

PYTHON?=python3.11
VENV=./env/bin
ROVERC=$(VENV)/python collect.py
GETCONF=$(ROVERC) get_config -c $(ROVER_CFG) -p

RADAR_SYS_IP=$(shell $(GETCONF) radar/args/capture/sys_ip)
RADAR_IF=$(shell $(GETCONF) radar/interface)
LIDAR_IF=$(shell $(GETCONF) lidar/interface)
RECV_BUF_SIZE=$(shell $(GETCONF) radar/args/capture/socket_buffer)

.PHONY: run server
run:
	echo "Launched with config: $(ROVER_CFG)"
	zellij --layout rover.kdl


server:
	cd server; ../env/bin/flask run --host=0.0.0.0

env:
	$(PYTHON) -m venv env
	$(VENV)/pip install -r requirements.txt


.PHONY: start, stop
start:
	$(ROVERC) start
stop:
	$(ROVERC) stop


# -- Configuration (on each reboot) -------------------------------------------

.PHONY: config radar-if radar-tty radar-socket imu-tty lidar-if
config: radar-if radar-tty radar-socket imu-tty
radar-if:
	sudo ifconfig $(RADAR_IF) $(RADAR_SYS_IP) netmask 255.255.255.0
radar-tty:
	sudo chmod 666 /dev/ttyACM0
imu-tty:
	sudo chmod 666 /dev/ttyUSB0
radar-socket:
	echo $(RECV_BUF_SIZE) | sudo tee /proc/sys/net/core/rmem_max
lidar-if:
	sudo ip addr add 169.254.11.103/16 dev $(LIDAR_IF)
