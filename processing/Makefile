#     ______  _____  _    _ _______  ______
#    |_____/ |     |  \  /  |______ |_____/
#    |    \_ |_____|   \/   |______ |    \_
#    Radar sensor fusion research platform
#

DATASET?=example


# ------------------------------ Python Config ------------------------------ #

ROVER_DIR=$(shell pwd)
VENV=./env/bin

env:
	$(PYTHON) -m venv env
	$(VENV)/pip install -r requirements.txt

# ------------------------------ Docker Config ------------------------------ #

DOCKER_IMAGE?=qoschatz/cartographer_ros
DATA_DIR?=$(ROVER_DIR)/data
ROVER_MNT?=/rover

# Docker run command
DOCKER=sudo docker run
DOCKER+=-it --privileged --net=host
DOCKER+=--env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all
DOCKER+=--env=DISPLAY --env=QT_X11_NO_MITSHM=1 --gpus all
DOCKER+=-v /tmp/.X11-unix:/tmp/.X11-unix
DOCKER+=-v $(ROVER_DIR):$(ROVER_MNT) -v $(DATA_DIR):/data
DOCKER+=$(DOCKER_IMAGE)

# docker_run is a function so we can add that pesky trailing (")
define docker_run
	$(DOCKER) bash -c "source $(ROVER_MNT)/catkin_ws/devel/setup.bash && $(1)"
endef

.PHONY: docker
docker:
	$(DOCKER) bash

# -------------------------- Docker / Cartographer -------------------------- #

# Docker paths
D_BAG_DIR=/data/$(DATASET)/_scratch
D_OUT_DIR=/data/$(DATASET)/_slam
D_PBSTREAM=$(D_OUT_DIR)/lidar.pbstream
D_BAG=/data/$(DATASET)/_scratch/lidar.bag

# Host paths
BAG_DIR=$(DATA_DIR)/$(DATASET)/_scratch
OUT_DIR=$(DATA_DIR)/$(DATASET)/_slam
PBSTREAM=$(OUT_DIR)/lidar.pbstream
POINTS=$(OUT_DIR)/lidar.bag_points.ply
TRAJ=$(OUT_DIR)/trajectory.csv

.PRECIOUS: $(PBSTREAM)
.PHONY: lidar validate info
lidar: $(TRAJ) $(POINTS)

trajectory: $(TRAJ)

validate:
	$(call docker_run, cartographer_rosbag_validate -bag_filename $(D_BAG))

info:
	$(call docker_run, rosbag info $(D_BAG))

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(PBSTREAM): $(OUT_DIR)
	$(call docker_run, roslaunch slam offline_cart_3d.launch \
		bag_filenames:=$(D_BAG))
	mv $(BAG_DIR)/lidar.bag.pbstream $(PBSTREAM)

$(POINTS): $(PBSTREAM)
	$(call docker_run, roslaunch slam assets_writer_cart_3d.launch \
		bag_filenames:=$(D_BAG) pose_graph_filename:=$(D_PBSTREAM))
	mv $(BAG_DIR)/lidar.bag_points.ply $(OUT_DIR)
	mv $(BAG_DIR)/lidar.bag_probability_grid.png $(OUT_DIR)
	mv $(BAG_DIR)/lidar.bag_xray* $(OUT_DIR)

$(BAG_DIR)/pose.bag: $(PBSTREAM)
	$(call docker_run, rosrun cartographer_ros \
		cartographer_dev_pbstream_trajectories_to_rosbag \
		--input $(D_PBSTREAM) --output $(D_BAG_DIR)/pose.bag)

$(TRAJ): $(BAG_DIR)/pose.bag
	$(call docker_run, rostopic echo -b $(D_BAG_DIR)/pose.bag -p trajectory_0 \
		> $(D_OUT_DIR)/trajectory.csv)
