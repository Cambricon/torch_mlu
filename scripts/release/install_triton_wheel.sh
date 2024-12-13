#!/bin/bash

PROPERTY_PATH=$(dirname $(readlink -f $0))
GENESIS_VERSION=$(cat "$PROPERTY_PATH/build.property" | grep "genesis" | awk -F '"' '{print $6}')
GENESIS_TAG=$(cat "$PROPERTY_PATH/build.property" | grep "genesis" | awk -F '"' '{print $4}')
TRITON_VERSION=$(cat "$PROPERTY_PATH/build.property" | grep "genesis" | awk -F '"' '{print $8}')
ARCH="manylinux2014_x86_64"
DISTRIBUTION=`uname -a`
if [[ "$DISTRIBUTION" == *"aarch64"* ]]; then
    ARCH="linux_aarch64"
fi
if [ $GENESIS_TAG != "release" ] ; then
    GENESIS_TAG=$GENESIS_VERSION
    GENESIS_VERSION=$(echo $GENESIS_TAG | cut -d '-' -f1)
fi

if [[ ${GENESIS_TAG} == "release" ]]; then
    wget -nv http://daily.software.cambricon.com/release/triton/${GENESIS_VERSION}/wheel/triton-${TRITON_VERSION}+mlu${GENESIS_VERSION}-cp310-cp310-${ARCH}.whl
    pip install triton-${TRITON_VERSION}+mlu${GENESIS_VERSION}-cp310-cp310-${ARCH}.whl
    rm triton-${TRITON_VERSION}+mlu${GENESIS_VERSION}-cp310-cp310-${ARCH}.whl
else
    wget -nv http://daily.software.cambricon.com/daily/triton/wheel/python3.10/master/${GENESIS_TAG}/triton-${TRITON_VERSION}+mlu${GENESIS_VERSION}-cp310-cp310-${ARCH}.whl
    pip install triton-${TRITON_VERSION}+mlu${GENESIS_VERSION}-cp310-cp310-${ARCH}.whl
    rm triton-${TRITON_VERSION}+mlu${GENESIS_VERSION}-cp310-cp310-${ARCH}.whl
fi
