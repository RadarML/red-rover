_ROVER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

alias roverc="$_ROVER_DIR/collect/env/bin/python $_ROVER_DIR/collect/collect.py"
alias roverp="$_ROVER_DIR/processing/env/bin/python $_ROVER_DIR/processing/process.py"
