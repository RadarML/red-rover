#     ______  _____  _    _ _______  ______
#    |_____/ |     |  \  /  |______ |_____/
#    |    \_ |_____|   \/   |______ |    \_
#    Radar sensor fusion research platform
#

PYTHON?=python3.11
JAX_CUDA?=cuda11_pip
ROVER_DIR=$(shell pwd)
ROVER_MNT?=/rover
DOCKER_IMAGE?=qoschatz/cartographer_ros

VENV=./env/bin

env:
	$(PYTHON) -m venv env
	$(VENV)/pip install --upgrade "jax[$(JAX_CUDA)]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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


DATASET?=example

BAG_DIR=$(shell realpath $(DATASET))/_scratch
BAG=$(BAG_DIR)/lidar.bag
OUT_DIR=$(shell realpath $(DATASET))/_slam

PBSTREAM=$(OUT_DIR)/lidar.pbstream
POINTS=$(OUT_DIR)_points.ply
POSE=$(BAG_DIR)/pose.bag
TRAJ=$(OUT_DIR)/trajectory.csv


.PRECIOUS: $(PBSTREAM)
.PHONY: lidar validate
lidar: $(POINTS) $(TRAJ)

trajectory: $(TRAJ)

validate:
	cartographer_rosbag_validate -bag_filename $(BAG)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)
	mkdir -p $(BAG_DIR)
	chmod 777 $(OUT_DIR)

$(PBSTREAM): $(OUT_DIR)
	-roslaunch slam offline_cart_3d.launch bag_filenames:=$(BAG)
	mv $(BAG_DIR)/lidar.bag.pbstream $(PBSTREAM)

$(POINTS): $(PBSTREAM)
	roslaunch slam assets_writer_cart_3d.launch \
		bag_filenames:=$(BAG) pose_graph_filename:=$(PBSTREAM)
	mv $(BAG_DIR)/lidar.bag_points.ply $(OUT_DIR)
	mv $(BAG_DIR)/lidar.bag_probability_grid.png $(OUT_DIR)
	mv $(BAG_DIR)/lidar.bag_xray* $(OUT_DIR)

$(POSE): $(PBSTREAM)
	rosrun cartographer_ros cartographer_dev_pbstream_trajectories_to_rosbag \
		--input $(PBSTREAM) --output $(POSE)

$(TRAJ): $(POSE)
	rostopic echo -b $(POSE) -p trajectory_0 > $(TRAJ)
