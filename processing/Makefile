PYTHON?=python3.11

VENV=./env/bin


env:
	$(PYTHON) -m venv env
	$(VENV)/pip install -r requirements.txt


docker-run:
	sudo docker run -it --privileged --net=host \
		--env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all \
		--env=DISPLAY --env=QT_X11_NO_MITSHM=1 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /home/tianshu/tmp:/test \
		qoschatz/cartographer_ros /bin/bash
