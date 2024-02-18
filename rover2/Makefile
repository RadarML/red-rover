PYTHON?=python3.11
RADAR_IF?=enx7cc2c6333f5f
RADAR_SYS_IP?=192.168.33.30

RADAR_NETMASK=255.255.255.0
VENV=./env/bin


env:
	$(PYTHON) -m venv env
	$(VENV)/pip install -r requirements.txt

.PHONY: radar-if
radar-if:
	sudo ifconfig $(RADAR_IF) $(RADAR_SYS_IP) netmask $(RADAR_NETMASK)
