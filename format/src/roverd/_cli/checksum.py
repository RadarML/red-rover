"""Compute checksums."""

import csv
import os
import subprocess
import tempfile
from fnmatch import fnmatch

from tqdm import tqdm


def _calculate_checksums(
    src: str, dst: str, patterns: list[str] | None = []
) -> None:
    print("Calculating checksums for:", src)
    paths = []
    for root, _, files in os.walk(src):
        for file in files:
            fullpath = os.path.join(root, file)
            rel_path = os.path.relpath(fullpath, src)
            if patterns is None or any(fnmatch(fullpath, p) for p in patterns):
                paths.append(rel_path)
    paths.sort()

    dst_dir = os.path.dirname(dst)
    if dst_dir and not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    with open(dst, 'w') as f:
        f.write("file,md5\n")
        for p in tqdm(paths, desc=src):
            file_path = os.path.join(src, p)
            result = subprocess.run(
                ['md5sum', file_path],
                capture_output=True, text=True, check=True)

            checksum = result.stdout.split()[0]
            f.write(f"{p},{checksum}\n")


def _compare_checksums(src: str, dst: str, patterns: list[str] | None) -> int:
    src_checksums = {}
    with open(src, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_checksums[row['file']] = row['md5']

    mismatched = 0
    with open(dst, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row['file']
            dst_md5 = row['md5']

            if patterns is not None:
                if not any(fnmatch(path, p) for p in patterns):
                    continue

            if path not in src_checksums:
                mismatched += 1
                print(f"Missing: {dst}/{path}")
            else:
                src_md5 = src_checksums[path]
                if src_md5 != dst_md5:
                    mismatched += 1
                    print(
                        f"Mismatch: {dst}/{path}; expected {src_md5}, "
                        f"actual {dst_md5}")

    return mismatched


def cli_checksum(
    src: str, dst: str, /, include: str | None = None
) -> int:
    r"""Compute and/or validate MD5 checksums for all data files in a trace.

    ```sh
    # Calculate
    uv run roverd checksum /path/to/data/trace /path/to/checksums/trace
    # Compare
    uv run roverd checksum /path/to/reference/trace /path/to/validate/trace
    # Validate
    uv run roverd checksum /path/to/checksums/trace /path/to/data_copy/trace
    ```

    !!! tip

        It may be helpful to chain `roverd list`:

        === "Calculate"

            ```sh
            for i in `roverd list ./iq1m`; do
                uv run roverd checksum ./iq1m/$i ./downloaded/$i;
            done
            ```

        === "Validate"

            ```sh
            for i in `roverd list ./iq1m`; do
                uv run roverd checksum ./reference/$i ./downloaded/$i;
            done
            ```

    This CLI tool can perform a number of actions depending on the input `src` and `dst`:

    | `src`     | `dst`     | Action |
    |-----------|-----------|--------|
    | Directory | Non-existent path | Compute checksums for all files in `src` and write to `dst`. |
    | Directory | File      | Compute checksums for all files in `src`, then compare `dst` against `src`. |
    | Directory | Directory | Compute checksums for all files in `src` and `dst`, then compare `dst` against `src`. |
    | File      | File      | Compare checksums in `dst` against those in `src`. |
    | File      | Directory | Compute checksums for all files in `dst`, then compare against those in `src`. |

    - Computed checksums are formatted as CSV file with two columns: `file`
        and `md5`, where `file` indicates the relative path of that data file
        to the trace root directory, and `md5` indicates the MD5 hash of the
        file (i.e., `md5sum <file>`, which excludes all metadata).
    - You can also supply a `filter`, which is a file path to a
        newline-separated list of glob patterns. Only files matching at least
        one of the patterns will be included in the checksum computation and/or
        verification.

    Args:
        src: path to the trace directory.
        dst: output checksum file.
        include: if specified, only compute checksums for and/or match glob
            patterns listed in this filter file.
    """
    patterns = None
    if include is not None:
        with open(include, 'r') as f:
            patterns = [line.strip() for line in f if line.strip()]

    # src is a directory => src points to trace data
    if os.path.isdir(src):
        # dst is a file or directory => comparison mode.
        if os.path.exists(dst):
            tmp_src = tempfile.NamedTemporaryFile(
                mode='w', suffix='.csv', delete=False)
            tmp_src.close()
            _calculate_checksums(src, tmp_src.name, patterns=patterns)
            src = tmp_src.name
        # else => calculation mode
        else:
            # calculate checksums for src into dst
            _calculate_checksums(src, dst, patterns=patterns)
            return 0  # nothing left to do

    # dst is a directory => dst points to trace data
    if os.path.isdir(dst):
        tmp_dst = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False)
        tmp_dst.close()
        _calculate_checksums(dst, tmp_dst.name, patterns=patterns)
        dst = tmp_dst.name

    mismatches = _compare_checksums(src, dst, patterns=patterns)
    return 1 if mismatches > 0 else 0
