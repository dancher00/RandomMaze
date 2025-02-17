#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 {q_learning|value_iteration|policy_iteration}"
    exit 1
fi

METHOD=$1
docker run -it -v "$(pwd)/results":/app/results randommaze python main.py --method "$METHOD"
