# Roverp: data processing

Install:

=== "pip + local"

    ```sh
    git clone git@github.com:WiseLabCMU/red-rover.git
    pip install -e ./red-rover/processing[graphics,pcd,ply,semseg]
    ```

=== "pip + github"

    ```sh
    pip install "roverp@git+ssh://git@github.com/WiseLabCMU/red-rover.git#subdirectory=processing"
    ```

=== "pip + github + extras"

    ```sh
    pip install "roverp[graphics,pcd,ply,semseg]@git+ssh://git@github.com/WiseLabCMU/red-rover.git#subdirectory=processing"
    ```

Extras:

- `graphics`
- `pcd`
- `ply`
- `semseg`

!!! info

    If you need to run cartographer SLAM, see the [dockerized cartographer setup instructions](docker.md).
