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


BAG?=lidar.bag

BAG_ABS=$(shell realpath $(BAG))
BAG_DIR=$(shell dirname $(BAG_ABS))
PBSTREAM=$(BAG_ABS).pbstream
POINTS=$(BAG_ABS)_points.ply
POSE=$(BAG_DIR)/pose.bag
TRAJ=$(BAG_DIR)/trajectory.csv


.PRECIOUS: $(PBSTREAM)
.PHONY: lidar validate
lidar: $(POINTS) $(TRAJ)

trajectory: $(TRAJ)

validate:
	cartographer_rosbag_validate -bag_filename $(BAG_ABS)

$(PBSTREAM):
	-roslaunch slam offline_cart_3d.launch bag_filenames:=$(BAG_ABS)

$(POINTS): $(PBSTREAM)
	roslaunch slam assets_writer_cart_3d.launch \
		bag_filenames:=$(BAG_ABS) pose_graph_filename:=$(PBSTREAM)

$(POSE): $(PBSTREAM)
	rosrun cartographer_ros cartographer_dev_pbstream_trajectories_to_rosbag \
		--input $(PBSTREAM) --output $(POSE)

$(TRAJ): $(POSE)
	rostopic echo -b $(POSE) -p trajectory_0 > $(TRAJ)
