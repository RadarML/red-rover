#     ______  _____  _    _ _______  ______
#    |_____/ |     |  \  /  |______ |_____/
#    |    \_ |_____|   \/   |______ |    \_
#    Radar sensor fusion research platform
#

ifndef ROVER_CFG
$(error No config file specified: ROVER_CFG is not set)
endif

VENV=./.venv/bin
ROVERC=$(VENV)/python cli.py
GETCONF=$(ROVERC) get-config --config $(ROVER_CFG) --path

RADAR_SYS_IP=$(shell $(GETCONF) radar/args/capture/sys_ip)
RADAR_IF=$(shell $(GETCONF) radar/interface)
LIDAR_IF=$(shell $(GETCONF) lidar/interface)
LIDAR_IP=$(shell ip -f inet addr show enp86s0 | grep 'inet ' | awk '{print $$2}')
RECV_BUF_SIZE=$(shell $(GETCONF) radar/args/capture/socket_buffer)

.PHONY: run server start stop
server:
	$(VENV)/flask run --host=0.0.0.0
run:
	echo "Launched with config: $(ROVER_CFG)"
	zellij --layout rover.kdl
start:
	$(ROVERC) start
stop:
	$(ROVERC) stop

lsconfig:
	$(VENV)/python config/summary.py

# -- Configuration (on each reboot) -------------------------------------------

.PHONY: config radar-if radar-tty radar-socket imu-tty lidar-if
config: radar-if radar-tty radar-socket imu-tty
radar-if:
	-sudo ifconfig $(RADAR_IF) $(RADAR_SYS_IP) netmask 255.255.255.0
radar-tty:
	-sudo chmod 666 /dev/ttyACM0
imu-tty:
	-sudo chmod 666 /dev/ttyUSB0
radar-socket:
	-echo $(RECV_BUF_SIZE) | sudo tee /proc/sys/net/core/rmem_max
lidar-if:
	-sudo ip addr add $(LIDAR_IP) dev $(LIDAR_IF)
