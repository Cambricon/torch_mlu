#!/bin/bash

# it is recommmended that a user sets the ${SRC_PACKAGES} and ${NEUWARE_HOME} in advance
# the assumption of this script is that pytorch, catch are all placed under ${SRC_PACKAGES}

# docker will have ${SRC_PACKAGES} set as /torch/src
if [ x"${SRC_PACKAGES}" == "x" ]; then
  export SRC_PACKAGES=$PWD/pytorch/src
  export TORCH_HOME=$PWD/pytorch/models/pytorch_models
else
  export TORCH_HOME=${SRC_PACKAGES}/pytorch_models
fi

export PYTORCH_HOME=${SRC_PACKAGES}/pytorch
export CATCH_HOME=${SRC_PACKAGES}/torch_mlu

if [ x"${NEUWARE_HOME}" == "x" ]; then
  export NEUWARE_HOME=usr/local/neuware
fi

# docker will already have $NEUWARE_HOME set as /torch/neuware_home
export PATH=$PATH:$NEUWARE_HOME/bin
export LD_LIBRARY_PATH=$NEUWARE_HOME/lib64:$LD_LIBRARY_PATH

# $PWD is used for backward compatibility, we will eventually replace it with ${SRC_PACKAGES} or something else once the
# use scenorios are fully identified
if [ -d ${PWD}/datasets ]; then
  export DATASET_HOME=$PWD/datasets
  export IMAGENET_PATH=$DATASET_HOME/imagenet/
  export VOC2012_PATH_PYTORCH=$DATASET_HOME/
  export VOC2007_PATH_PYTORCH=$DATASET_HOME/
  export COCO_PATH_PYTORCH=$DATASET_HOME/
  export FDDB_PATH_PYTORCH=$DATASET_HOME/
  export IMAGENET_PATH_PYTORCH=$DATASET_HOME/
  export ICDAR_PATH_PYTORCH=$DATASET_HOME/ICDAR_2015/test_img
  export VOC_DEVKIT=$DATASET_HOME/
fi

export GLOG_alsologtostderr=true
# Set log level which is output to stderr, 0: INFO/WARNING/ERROR/FATAL, 1: WARNING/ERROR/FATAL, 2: ERROR/FATAL, 3: FATAL,
export GLOG_minloglevel=0
