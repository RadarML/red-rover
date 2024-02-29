PYTHON?=python3.11
RADAR_IF?=enx207bd2be1456
RADAR_SYS_IP?=192.168.33.30
RECV_BUF_SIZE?=2048000

RADAR_NETMASK=255.255.255.0
VENV=./env/bin


.PHONY: run
run:
	zellij --layout rover.kdl

env:
	$(PYTHON) -m venv env
	$(VENV)/pip install -r requirements.txt

.PHONY: config radar-if radar-tty radar-socket imu-tty
config: radar-if radar-tty radar-socket imu-tty
radar-if:
	sudo ifconfig $(RADAR_IF) $(RADAR_SYS_IP) netmask $(RADAR_NETMASK)

radar-tty:
	sudo chmod 666 /dev/ttyACM0

imu-tty:
	sudo chmod 666 /dev/ttyUSB0

radar-socket:
	echo $(RECV_BUF_SIZE) | sudo tee /proc/sys/net/core/rmem_max
