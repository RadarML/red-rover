PYTHON?=python3.11
ROVER_DIR=$(shell pwd)
ROVER_MNT?=/rover
DOCKER_IMAGE?=qoschatz/cartographer_ros

VENV=./env/bin


env:
	$(PYTHON) -m venv env
	$(VENV)/pip install -r requirements.txt


.PHONY: docker-run
docker-run:
	sudo docker run -it --privileged --net=host \
		--env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all \
		--env=DISPLAY --env=QT_X11_NO_MITSHM=1 \
		--gpus all \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(ROVER_DIR):$(ROVER_MNT) \
		$(DOCKER_IMAGE) /bin/bash
