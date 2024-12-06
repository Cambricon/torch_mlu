#!/bin/bash
set -e
PACKAGE_NAME="torch"
PACKAGE_PATH=$(pip show "$PACKAGE_NAME" 2>/dev/null | grep -i "^Location:" | awk '{print $2}')
export LD_LIBRARY_PATH="${PACKAGE_PATH}/${PACKAGE_NAME}/lib":$LD_LIBRARY_PATH

./torch_mlu_demo ../add_model.pt 4 1000
./torch_mlu_demo ../conv_trace_model.pt 4 1000
