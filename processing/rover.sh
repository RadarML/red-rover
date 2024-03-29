_ROVER_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export ROVERP="$_ROVER_DIR/env/bin/python $_ROVER_DIR/process.py"
alias roverp=$ROVERP