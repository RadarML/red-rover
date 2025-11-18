# Download

!!! abstract

    The preferred download method is via the [Open Storage Network](https://openstoragenetwork.github.io/) (OSN), which is a NSF-funded distributed data sharing and transfer service based on S3 buckets. We currently have a storage allocation on OSN which we are trialing to distribute I/Q-1M.

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

    ```toml
    [osn-ro]
    type = s3
    provider = Ceph
    access_key_id = # ask tianshu or get this from coldfront
    secret_access_key =   # ...
    endpoint = https://uri.osn.mghpcc.org
    no_check_bucket = true
    ```


??? info "Test the configuration"

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

To download the full dataset (3.2TB / 2.9TiB), use:
```sh
rclone sync osn-ro:/cmu-wiselab-iq1m/data ./iq1m -v
```

- When using `rclone sync`, you can stop (interrupt with ctrl+C) and resume downloading at any time.

!!! warning "Symlink Videos"

    The videos distributed via OSN have been anonymized to blur out faces; you can symlink these in place of the raw videos:
    ```sh
    for i in `uv run roverd list /path/to/dataset`; do
        ln -s /path/to/dataset/$i/_camera/video.avi /path/to/dataset/$i/camera/video.avi;
    done
    ```

??? info "Download Traces Individually"

    You can also download each setting or trace separately:
    ```sh
    rclone sync osn-ro:/cmu-wiselab-iq1m/data/indoor ./iq1m/indoor -v
    rclone sync osn-ro:/cmu-wiselab-iq1m/data/indoor/baker ./iq1m/indoor/baker -v
    rclone sync osn-ro:/cmu-wiselab-iq1m/data/indoor/baker/baker.1.fwd ./iq1m/indoor/baker/baker.1.fwd -v
    ```

??? info "Exclude Unneeded Files"

    You can exclude certain files (e.g., lidar reflectance / NIR) using
    ```sh
    rclone sync osn-ro:/cmu-wiselab-iq1m/data ./iq1m -v --exclude */lidar/rfl */lidar/nir
    ```

    See the `rclone sync` [documentation](https://rclone.org/commands/rclone_sync) for more details and other options.


## Upload

!!! info

    We currently have a 5TB bucket provisioned in OSN. To check the current utilization:
    ```sh
    rclone size osn-rw:/cmu-wiselab-iq1m
    ```   
    
Create a read/write configuration:

```toml
[osn-rw]
type = s3
provider = Ceph
access_key_id = # ask tianshu or get this from coldfront
secret_access_key =   # ...
endpoint = https://uri.osn.mghpcc.org
no_check_bucket = true
```

Then, upload with
```sh
rclone sync /data osn-rw:/cmu-wiselab-iq1m/data --include-from iq1m/upload.txt -v
```

The original instructions from OSN can also be found [here](https://openstoragenetwork.github.io/docs/dataset-access/rclone/).

??? quote "Uploaded Files"

    `iq1m.txt` includes the following:
    ```
    --8<-- "iq1m/upload.txt"
    ```

## FTP Fallback

If you are unable to download the dataset via OSN, we have set up a "fallback" FTP server which also hosts a copy of the dataset.

!!! warning

    This FTP server is hosted in our lab, and has limited resources compared to OSN; to avoid interfering with other traffic, the server is also rate-limited to 100Mbps. **Please only use this option if OSN is down or otherwise inaccessible.**

You can download files from the FTP server using any standard FTP client, e.g., [FileZilla](https://filezilla-project.org/).

| URL | Port |
|-----|------|
| arena-gw.lan.cmu.edu | 37285 |

!!! warning

    The FTP server address and port are subject to change, and our lab network is explicitly not a high-availability service!

## Verify Files

After downloading the dataset, you may wish to verify the integrity of the files by comparing checksums.

!!! info

    You can find the reference checksum files in the [red-rover repository](https://github.com/RadarML/red-rover/blob/main/iq1m/checksums.tar.gz). See [`roverd checksum`](../roverd/cli.md) for more details about how these checksums are computed and unformatted; the format should be fairly self-explanatory once you `untar` the files.

Assuming that you've [installed roverd](../roverd/index.md) into your environment:

1. Calculate checksums on your downloaded copy:

    ```sh
    uv run roverd checksum /path/to/downloaded/iq1m /path/to/checksums/output
    ```

2. Compare the calculated checksums against the reference checksums:

    ```sh
    uv run roverd checksum-compare /path/to/original/checksums /path/to/checksums/output
    ```
