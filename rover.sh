_ROVER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "$#" -eq 0 ]; then
    alias roverp="$_ROVER_DIR/processing/env/bin/python $_ROVER_DIR/processing/process.py"

    echo "ROVER_CFG not provided; only roverp will be available."
else
    alias roverc="$_ROVER_DIR/collect/env/bin/python $_ROVER_DIR/collect/collect.py"
    alias roverp="$_ROVER_DIR/processing/env/bin/python $_ROVER_DIR/processing/process.py"
    export ROVER_CFG=$(realpath $1)

    echo "Initialized rover with config: $ROVER_CFG"
fi
