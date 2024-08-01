Data Processing Pipeline
========================

.. image:: processing.svg
   :alt: Data processing pipeline

Common Recipes
--------------

Get dataset metadata::

    roverp info -p path/to/dataset
    
Upload data to storage server or external drive::

    roverp export -p {path/to/dataset} -o {path/to/destination}

Prepare radarhd data::

    export SRC=path/to/dataset
    export DST=path/to/destination

    roverp export -p $SRC -o $DST --metadata
    roverp align -p $DST --mode left
    roverp decompress -p $SRC -o $DST
    cp $SRC/radar/iq $DST/radar/iq

Prepare DART data::

    export DATASET=dataset
    roverp rosbag -p data/$DATASET
    make lidar
    roverp sensorpose -p data/$DATASET -s radar
    roverp fft -p data/$DATASET --mode hybrid

Generate reports (requires DART data)::

    export DATASET=path/to/dataset
    roverp report -p $DATASET
    roverp video -p $DATASET

Generate simulations (requires DART data)::

    export DATASET=path/to/dataset
    roverp nearest -p $DATASET
    roverp lidarmap -p $DATASET
    roverp simulate -p $DATASET

List all datasets in current directory::

    find -name 'config.yaml' | xargs dirname


Makefile
--------
The supplied makefile automates the following steps:

- `make trajectory`: run cartographer.
    - Specify the target dataset with `DATASET=path/to/dataset`; all datasets are assumed to be placed inside a `red-rover/process/data` directory.
    - Optionally specify `HEADLESS=1` to run in headless mode (without a rviz window). When running in headless mode, the ROS node will automatically exit when complete; otherwise, an rviz window will open, and the process will wait for the window to be closed before continuing.
- `make lidar`: run cartographer and generate pointcloud outputs.
    - This includes `make trajectory`, and has the same arguments.
- `make validate`, `make info`: validate, get info for rosbag generated with `roverp rosbag`.


Scripts
-------

Scripts are categorized as follows:

- **Convert**: convert data format for compatibility with external software.
- **Create**: generate a specific representation based on the input data.
- **Export**: convert or copy data for training or other distribution.
- **Get**: print or visualize dataset metadata.
- **Render**: create a video visualization of the data.
- **Run**: a compute (GPU) intensive operation to apply a given algorithm.

.. toctree::
   :maxdepth: 2

   scripts/convert.rst
   scripts/create.rst
   scripts/export.rst
   scripts/get.rst
   scripts/render.rst
   scripts/run.rst
