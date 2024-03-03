PYTHON?=python3.11

VENV=./env/bin
ROVERC=$(VENV)/python collect.py

RADAR_SYS_IP=$(shell $(ROVERC) get_config -p radar/args/capture/sys_ip)
RADAR_IF=$(shell $(ROVERC) get_config -p radar/interface)
RECV_BUF_SIZE=$(shell $(ROVERC) get_config -p radar/args/capture/socket_buffer)
RADAR_NETMASK=255.255.255.0

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

.PHONY: start, stop
start:
	$(ROVERC) start

stop:
	$(ROVERC) stop
