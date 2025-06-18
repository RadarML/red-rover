# Set `roverp` as `process.py` in the source directory
_ROVER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# `$ROVERP` provides a link which can be used when aliases are not available,
# e.g. in scripts or `nq`.
export ROVERP="$_ROVER_DIR/.venv/bin/roverp"
# `roverp` provides a slightly more convenient alias.
alias roverp=$ROVERP

# Set up docker GUI (and ignore the output)
xhost +local:docker > /dev/null
