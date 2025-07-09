# Open Storage Network

!!! abstract

    The [Open Storage Network](https://openstoragenetwork.github.io/) (OSN) is a NSF-funded distributed data sharing and transfer service based on S3 buckets. We currently have a storage allocation on OSN which we are trialing to distribute I/Q-1M.

!!! tip "OSN is on Internet2"

    OSN is connected to [internet2](https://internet2.edu/); institutions connected to internet2 can benefit from high download speeds.

    In our benchmarks from CMU campus, we were able to achieve ~600mbps download and ~1500mbps upload speeds.

## Set up `rclone`

The best way to interact with OSN is through [rclone](https://rclone.org/docs/).

!!! info

    The I/Q-1m dataset is currently in pre-release, and is not accessible to the general public.

    Ask Tianshu (<tianshu2@andrew.cmu.edu>) for the access keys, or find them on the allocations page in [OSN ColdFront](https://coldfront.osn.mghpcc.org/) if you have been added to the allocation as a user/manager.

After installing `sudo apt-get install rclone`, you will need to create a configuration with our bucket information:

- Find the config file name using `rclone config file`.

    !!! info
        
        The config path should be something like `~/.config/rclone/rclone.config` i.e., `/home/<user>/.config/rclone/rclone.config`.

- Add the following to the file (create the file if it doesn't already exist):

    === "Read Only"

        ```toml
        [osn-ro]
        type = s3
        provider = Ceph
        access_key_id = # ask tianshu or get this from coldfront
        secret_access_key =   # ...
        endpoint = https://uri.osn.mghpcc.org
        no_check_bucket = true
        ```

    === "Read/Write"

        ```toml
        [osn-rw]
        type = s3
        provider = Ceph
        access_key_id = # ask tianshu or get this from coldfront
        secret_access_key =   # ...
        endpoint = https://uri.osn.mghpcc.org
        no_check_bucket = true
        ```

To test this configuration:

=== "List files"

    ```sh
    $ rclone lsd osn-ro:/cmu-wiselab-iq1m
            0 2025-06-25 16:24:51        -1 data
    ```

    `lsd` lists all directories. You can also use:

    - `rclone ls`: list all files, recursively
    - `rclone ls --max-depth 1`: list only files in the specified folder

=== "Download a sample file"

    ```sh
    rclone copy osn-ro:/cmu-wiselab-iq1m/data/bike/bloomfield.back/config.yaml .
    ```

## Download

To download the full dataset, use:
```sh
rclone sync osn-ro:/cmu-wiselab-iq1m/data ./iq1m -v
```

- When using `rclone sync`, you can stop (interrupt with ctrl+C) and resume downloading at any time.

You can also download each setting or trace separately:
```sh
rclone sync osn-ro:/cmu-wiselab-iq1m/data/indoor ./iq1m/indoor -v
rclone sync osn-ro:/cmu-wiselab-iq1m/data/indoor/baker ./iq1m/indoor/baker -v
rclone sync osn-ro:/cmu-wiselab-iq1m/data/indoor/baker/baker.1.fwd ./iq1m/indoor/baker/baker.1.fwd -v
```

!!! tip

    You can exclude certain files (e.g., lidar reflectance / NIR) using
    ```sh
    rclone sync osn-ro:/cmu-wiselab-iq1m/data ./iq1m -v --exclude */lidar/rfl */lidar/nir
    ```

    See the `rclone sync` [documentation](https://rclone.org/commands/rclone_sync) for more details and other optoins.


!!! warning

    The camera files (`camera/video.avi`) are currently not included, since we still need to build the anonymization pipeline; for now, please contact Tianshu (<tianshu2@andrew.cmu.edu>) for access.

## Upload

!!! info

    We currently have a 5TB bucket provisioned in OSN. To check the current utilization:
    ```sh
    rclone size osn-rw:/cmu-wiselab-iq1m
    ```   
    
Assuming you have set up the `osn-rw` read/write configuration with the appropriate keys, upload with
```sh
rclone sync /data osn-rw:/cmu-wiselab-iq1m/data --include-from iq1m/upload.txt -v
```

The original instructions from OSN can also be found [here](https://openstoragenetwork.github.io/docs/dataset-access/rclone/).

??? quote "Uploaded Files"

    `iq1m.txt` includes the following:
    ```
    */config.yaml
    */_camera/meta.json
    */_camera/pose.npz
    */_camera/segment
    */_camera/segment_i
    */_camera/ts
    */camera/meta.json
    */camera/ts
    */imu/acc
    */imu/avel
    */imu/meta.json
    */imu/rot
    */imu/ts
    */_lidar/pose.npz
    */lidar/lidar.json
    */lidar/meta.json
    */lidar/nir
    */lidar/nir_i
    */lidar/rfl
    */lidar/rfl_i
    */lidar/rng
    */lidar/rng_i
    */lidar/ts
    */_radar/pose.npz
    */radar/iq
    */radar/meta.json
    */radar/radar.json
    */radar/ts
    */radar/valid
    */_slam/trajectory.csv
    ```
