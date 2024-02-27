PYTHON?=python3.11
RADAR_IF?=enx207bd2be1456
RADAR_SYS_IP?=192.168.33.30

RADAR_NETMASK=255.255.255.0
VENV=./env/bin


.PHONY: run
run:
	zellij --layout rover.kdl

env:
	$(PYTHON) -m venv env
	$(VENV)/pip install -r requirements.txt

.PHONY: config radar-if radar-tty imu-tty
config: radar-if radar-tty imu-tty
radar-if:
	sudo ifconfig $(RADAR_IF) $(RADAR_SYS_IP) netmask $(RADAR_NETMASK)

radar-tty:
	sudo chmod 666 /dev/ttyACM0

imu-tty:
	sudo chmod 666 /dev/ttyUSB0
