#!/bin/bash
set -e
ABS_DIR_PATH=$(dirname $(readlink -f $0))
TORCH_MLU_PATH=$ABS_DIR_PATH/../../

DOCKER_FILE=unset
IMAGE_NAME=unset
TAG=unset

PYTHON_VERSION="3.10"

while getopts "f:n:t:" opt
do
  case ${opt} in
    f)
        DOCKER_FILE=${OPTARG};;
    n)
	      IMAGE_NAME=$OPTARG;;
    t)
	      TAG=$OPTARG;;
    ?)
      echo "There is unrecognized parameter."
      exit 1;
  esac
done

build_base_docker(){
  build_base_cmd="docker build --no-cache --network=host --rm \
                  -t ${IMAGE_NAME}:${TAG} -f ${DOCKER_FILE} ."
  echo "build base command : "${build_base_cmd}
  eval ${build_base_cmd}
}

echo "==========================================="
echo "DOCKER FILE : "${DOCKER_FILE}
echo "IMAGE NAME : "${IMAGE_NAME}
echo "TAG : "${TAG}
#echo "PYTHON_VERSION : "${PYTHON_VERSION}
build_base_docker
echo "==========================================="
