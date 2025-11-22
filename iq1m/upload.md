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
