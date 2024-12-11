#!/bin/bash
set -e

ABS_DIR_PATH=$(dirname $(readlink -f $0))
TORCH_MLU_PATH=$ABS_DIR_PATH/../../
RELEASE_TYPE=NULL
BRANCH=NULL   # resolve for future use
PYTORCH_VERSION="release/2.4" # set to native release version(eg: 2.1.0, 2.2.0) or branch(eg: release/2.1, release/2.2) in torch_mlu develop branch
TORCH_MLU_BRANCH='r2.4_develop'
TORCHAUDIO_MLU_BRANCH='r2.4_develop'
VISION_VERSION='release/0.19'  # set to native release version(eg: 1.6.0) in torch_mlu develop branch
AUDIO_VERSION='release/2.4'  # set to native release version(eg: 2.1.0) in torch_mlu develop branch
PYTORCH_MODELS_BRANCH="main"
BUILDER_DOCKER="yellow.hub.cambricon.com/pytorch/base/x86_64/manylinux:native-manylinux-builder-cpu-main"
DRIVER_VERSION=NULL
CNTOOLKIT_VERSION=NULL
CNNL_VERSION=NULL
CNNLExtra_VERSION=NULL
CNCL_VERSION=NULL
CNDALI_VERSION=NULL
CNCV_VERSION=NULL
GENESIS_TAG=NULL
GENESIS_VERSION=NULL
TRITON_VERSION=NULL
MAGICMIND_VERSION=NULL
OS_TYPE=NULL
OS_VERSION=NULL
IMAGE_WHEEL_NAME=pytorch_manylinux
IMAGE_DOCKER_NAME=u20_yy
TAG=wheel_py3108
DOCKER_FILE=NULL
PACKAGE_ARCH="x86_64"
LIBRARIES_LIST="cncv cncl cnpapi"
PLATFORM="amd64"
ABI_VERSION="old"
PY_SUFFIX="py310"
PYTHON_VERSION="3.10"
TORCH_MLU_COMMIT_ID=NULL

while getopts "r:b:c:e:p:n:x:y:z:o:v:w:d:t:f:a:m:" opt
do
  case $opt in
    r)
	    RELEASE_TYPE=$OPTARG;;
    b)
        BRANCH=$OPTARG;;
    c)
        TORCH_MLU_BRANCH=$OPTARG;;
    e)
        TORCHAUDIO_MLU_BRANCH=$OPTARG;;
    l)
        CNDALI_VERSION=$OPTARG;;
    p)
        PYTORCH_MODELS_BRANCH=$OPTARG;;
    o)
	    OS_TYPE=$OPTARG;;
    v)
        OS_VERSION=$OPTARG;;
    d)
	    IMAGE_DOCKER_NAME=$OPTARG;;
    t)
	    TAG=$OPTARG;;
    f)
	    DOCKER_FILE=$OPTARG;;
    a)
        PACKAGE_ARCH=$OPTARG;;
    i)
        ABI_VERSION=$OPTARG;;
    m)
        TORCH_MLU_COMMIT_ID=$OPTARG;;
    ?)
	      echo "there is unrecognized parameter."
	      exit 1;;
  esac
done

if [ ${PACKAGE_ARCH} == "aarch64" ];then
    BUILDER_DOCKER="yellow.hub.cambricon.com/pytorch/pytorch:v1.23.1-torch2.4.0-runtime-anolis8.8-py310"
fi

echo "======================================="
echo "RELEASE_TYPE: "$RELEASE_TYPE
echo "PYTORCH_VERSION: "$PYTORCH_VERSION
echo "PYTORCH_MODELS_BRANCH: "$PYTORCH_MODELS_BRANCH
echo "OS_TYPE: "$OS_TYPE
echo "OS_VERSION: "$OS_VERSION
echo "IMAGE_DOCKER_NAME: "$IMAGE_DOCKER_NAME
echo "TAG: "$TAG
echo "DOCKER_FILE: "$DOCKER_FILE
echo "PYTHON_VERSION: "$PYTHON_VERSION
echo "========================================"

read_ver_func() {
  python3 $ABS_DIR_PATH/json_parser.py
  output='dependency.txt'
  release='release.txt'
  RELEASE_VERSION=`cat $release | awk -F ':' '{print $3}'`
  echo "RELEASE_VERSION: $RELEASE_VERSION"

  # dep-package-version
  PACKAGE_MODULES=`cat $output | awk -F ':' '{print $1}'`
  echo "PACKAGE_MODULES: ${PACKAGE_MODULES}"

  PACKAGE_BRANCH=`cat $output | awk -F ':' '{print $2}'`
  echo "PACKAGE_BRANCH: $PACKAGE_BRANCH"

  PACKAGE_MODULE_VERS=`cat $output | awk -F ':' '{print $3}'`
  echo "PACKAGE_MODULE_VERS: $PACKAGE_MODULE_VERS"

  rm ${output}
  rm ${release}
}

fetch_neuware_ver_func() {
  sleep 30
  DRIVER_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "driver_requires" | awk -F '"' '{print $6}')
  CNTOOLKIT_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "cntoolkit" | awk -F '"' '{print $6}')
  # Here we use head -1 to merely get the entry with the whole word "cnnl". Otherwise, both the version of cnnl and cnnl_extra will be greped.
  CNNL_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep -w "cnnl" | awk -F '"' '{print $6}')
  CNNLExtra_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "cnnlextra" | awk -F '"' '{print $6}')
  CNCL_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "cncl" | awk -F '"' '{print $6}')
  CNCV_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "cncv" | awk -F '"' '{print $6}')
  CNDALI_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "cndali" | awk -F '"' '{print $6}')
  MLUOP_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep -w "mluop" | awk -F '"' '{print $6}')
  BENCHMARK_VERSION=benchmark_$(cat "$ABS_DIR_PATH/build.property" | grep "benchmark" | awk -F '"' '{print $6}')
  GENESIS_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "genesis" | awk -F '"' '{print $6}')
  GENESIS_TAG=$(cat "$ABS_DIR_PATH/build.property" | grep "genesis" | awk -F '"' '{print $4}')
  TRITON_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "genesis" | awk -F '"' '{print $8}')
  if [ $GENESIS_TAG != "release" ] ; then
      GENESIS_TAG=$GENESIS_VERSION
      GENESIS_VERSION=$(echo $GENESIS_TAG | cut -d '-' -f1)
  fi
  # MAGICMIND_VERSION=$(cat "$ABS_DIR_PATH/build.property" | grep "magicmind" | awk -F '"' '{print $6}')
}

fetch_cn_dep_func() {
  echo "independent_build.sh located in "$ABS_DIR_PATH

  PACKAGE_SERVER="http://daily.software.cambricon.com"
  PACKAGE_OS="Linux"

  arr_modules=(`echo $PACKAGE_MODULES`)
  arr_branch=(`echo $PACKAGE_BRANCH`)
  arr_vers=(`echo $PACKAGE_MODULE_VERS`)

  n=${#arr_vers[@]}
  echo "number of dependency: $n"

  # set download dir
  PACKAGE_DOWNLOAD_DIR="dep_libs_download"
  if [ ! -d $PACKAGE_DOWNLOAD_DIR ]; then
    mkdir $PACKAGE_DOWNLOAD_DIR
  else
    rm -rf $PACKAGE_DOWNLOAD_DIR/*
  fi

  # set extract dir
  PACKAGE_EXTRACT_DIR="dep_libs_extract"
  if [ ! -d ${PACKAGE_EXTRACT_DIR} ]; then
    mkdir ${PACKAGE_EXTRACT_DIR}
  else
    rm -rf ${PACKAGE_EXTRACT_DIR}/*
  fi

  NEUWARE_HOME="neuware_home"
  if [ ! -d ${NEUWARE_HOME} ]; then
    mkdir ${NEUWARE_HOME}
  else
    rm -rf ${NEUWARE_HOME}/*
  fi

  if [ -f "/etc/os-release" ]; then
    source /etc/os-release
    if [ ${OS_TYPE} == "NULL" ] || [ ${OS_VERSION} == "NULL" ];then
      OS_TYPE=${ID}
      OS_VERSION=${VERSION_ID}
      echo "[NOTICE] Use current OS ${OS_TYPE}:${OS_VERSION} ......"
    else
      echo "[NOTICE] Manual Specify OS ${OS_TYPE}:${OS_VERSION} ......"
    fi

    if [ ${OS_TYPE} == "ubuntu" ] || [ ${OS_TYPE} == "debian" ]; then
      for (( i =0; i < ${n}; i++))
      do
        if [ ${OS_TYPE} == "ubuntu" ]; then
          PACKAGE_DIST="Ubuntu"
        elif [ ${OS_TYPE} == "debian" ]; then
          PACKAGE_DIST="Debian"
        fi

        PACKAGE_DIST_VER=${OS_VERSION}

        if [ ${arr_modules[$i]} == "cntoolkit" ]; then
          PACKAGE_FILE=${arr_modules[$i]}"_"${arr_vers[$i]}"."${OS_TYPE}${OS_VERSION}"_${PLATFORM}.deb"
          echo "PACKAGE_FILE: $PACKAGE_FILE"
          PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"${PACKAGE_FILE}
          echo "PACKAGE_PATH: $PACKAGE_PATH"
          MODULE_PURE_VER=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
          wget ${PACKAGE_PATH} -P ${PACKAGE_DOWNLOAD_DIR} -nv
          # extract in extract dir
          pushd ${PACKAGE_EXTRACT_DIR}
          dpkg -x ../${PACKAGE_DOWNLOAD_DIR}/$PACKAGE_FILE ../${PACKAGE_DOWNLOAD_DIR}
          for filename in ../${PACKAGE_DOWNLOAD_DIR}/var/${arr_modules[$i]}"-"$MODULE_PURE_VER/*.deb; do
            if [[ "$filename" == *"edge"* ]]; then
              echo "skip edge package $filename"
              continue
            fi
            dpkg -x $filename . &
          done
          popd
        else
          if [[ ${arr_branch[$i]} == "daily" ||  ${arr_branch[$i]} == "temp" ]];then
            VERSION_FILE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/version.txt"
            wget $VERSION_FILE_PATH -P ${PACKAGE_DOWNLOAD_DIR} -nv
            PACKAGE_FILE=$(cat ${PACKAGE_DOWNLOAD_DIR}"/version.txt" | awk -F '=' '{if($1=="file") print $2}' | tr -d '\r')
            rm ${PACKAGE_DOWNLOAD_DIR}"/version.txt"
          else
            PACKAGE_FILE=${arr_modules[$i]}"_"${arr_vers[$i]}"."${OS_TYPE}${OS_VERSION}"_${PLATFORM}.deb"
          fi
          if [[ ${arr_branch[$i]} != "daily" && ${arr_modules[$i]} == "mluops" ]]; then
            PACKAGE_FILE=${arr_modules[$i]}"_"${arr_vers[$i]}"."${OS_TYPE}${OS_VERSION}"_"${PLATFORM}".deb"
          fi

          echo "PACKAGE_FILE: $PACKAGE_FILE"
          PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"${PACKAGE_FILE}
          echo "PACKAGE_PATH: $PACKAGE_PATH"
          wget ${PACKAGE_PATH} -P ${PACKAGE_DOWNLOAD_DIR} -nv
          # extract in extract dir
          pushd ${PACKAGE_EXTRACT_DIR}
          dpkg -x ../${PACKAGE_DOWNLOAD_DIR}/$PACKAGE_FILE . &
          popd
        fi
      done

      # wait for subprocesses
      wait
    elif [ ${OS_TYPE} == "centos" ]; then
      for (( i =0; i < ${n}; i++))
      do
        PACKAGE_DIST="CentOS"
        PACKAGE_DIST_VER=${OS_VERSION}
        if [ ${arr_modules[$i]} == "cntoolkit" ]; then
          PACKAGE_FILE=${arr_modules[$i]}"-"${arr_vers[$i]}".el${PACKAGE_DIST_VER}.x86_64.rpm"
          echo "PACKAGE_FILE: $PACKAGE_FILE"
          PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"${PACKAGE_FILE}
          echo "PACKAGE_PATH: $PACKAGE_PATH"
          MODULE_PURE_VER=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
          wget ${PACKAGE_PATH} -P ${PACKAGE_DOWNLOAD_DIR} -nv
          # extract in extract dir
          pushd ${PACKAGE_DOWNLOAD_DIR}
          rpm2cpio $PACKAGE_FILE | cpio -div
          popd
          pushd ${PACKAGE_EXTRACT_DIR}
          for filename in ../${PACKAGE_DOWNLOAD_DIR}/var/${arr_modules[$i]}"-"$MODULE_PURE_VER/*.rpm; do
            if [[ "$filename" == *"edge"* ]]; then
              echo "skip edge package $filename"
              continue
            fi
            rpm2cpio $filename | cpio -div
          done
          popd
        else
          if [[ ${arr_branch[$i]} == "daily" ||  ${arr_branch[$i]} == "temp" ]];then
            VERSION_FILE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/version.txt"
            wget $VERSION_FILE_PATH -P ${PACKAGE_DOWNLOAD_DIR} -nv
            PACKAGE_FILE=$(cat ${PACKAGE_DOWNLOAD_DIR}"/version.txt" | awk -F '=' '{if($1=="file") print $2}')
            rm ${PACKAGE_DOWNLOAD_DIR}"/version.txt"
          else
            PACKAGE_FILE=${arr_modules[$i]}"-"${arr_vers[$i]}".el${PACKAGE_DIST_VER}.x86_64.rpm"
          fi
          if [[ ${arr_branch[$i]} != "daily" && ${arr_modules[$i]} == "mluops" ]]; then
            PACKAGE_FILE=${arr_modules[$i]}"-"${arr_vers[$i]}".el"${PACKAGE_DIST_VER}".x86_64.rpm"
          fi
          echo "PACKAGE_FILE: $PACKAGE_FILE"
          PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"${PACKAGE_FILE}
          echo "PACKAGE_PATH: $PACKAGE_PATH"
          wget ${PACKAGE_PATH} -P ${PACKAGE_DOWNLOAD_DIR} -nv
          pushd ${PACKAGE_EXTRACT_DIR}
          rpm2cpio ../${PACKAGE_DOWNLOAD_DIR}/$PACKAGE_FILE | cpio -div
          popd
        fi
      done
    elif [ ${OS_TYPE} == "kylin" ]; then
      for (( i =0; i < ${n}; i++))
      do
        PACKAGE_DIST="Kylin"
        PACKAGE_DIST_VER=${OS_VERSION}
        if [ ${arr_modules[$i]} == "cntoolkit" ]; then
          PACKAGE_FILE=${arr_modules[$i]}"-"${arr_vers[$i]}".ky10.x86_64.rpm"
          echo "PACKAGE_FILE: $PACKAGE_FILE"
          PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"${PACKAGE_FILE}
          echo "PACKAGE_PATH: $PACKAGE_PATH"
          MODULE_PURE_VER=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
          wget ${PACKAGE_PATH} -P ${PACKAGE_DOWNLOAD_DIR}
          # extract in extract dir
          pushd ${PACKAGE_DOWNLOAD_DIR}
          rpm2cpio $PACKAGE_FILE | cpio -div
          popd
          pushd ${PACKAGE_EXTRACT_DIR}
          for filename in ../${PACKAGE_DOWNLOAD_DIR}/var/${arr_modules[$i]}"-"$MODULE_PURE_VER/*.rpm; do
            if [[ "$filename" == *"edge"* ]]; then
              echo "skip edge package $filename"
              continue
            fi
            rpm2cpio $filename | cpio -div
          done
          popd
        else
          if [[ ${arr_modules[$i]} == "cnnl" && ${arr_branch[$i]} == "temp" ]];then
    	    VERSION_FILE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/version.txt"
            wget $VERSION_FILE_PATH -P ${PACKAGE_DOWNLOAD_DIR}
            PACKAGE_FILE=$(cat ${PACKAGE_DOWNLOAD_DIR}"/version.txt" | awk -F '=' '{if($1=="file") print $2}')
          else
            PACKAGE_FILE=${arr_modules[$i]}"-"${arr_vers[$i]}".ky10.x86_64.rpm"
          fi
          if [[ ${arr_modules[$i]} == "mluops" ]]; then
            PACKAGE_FILE=${arr_modules[$i]}"-"${arr_vers[$i]}".ky10.x86_64.rpm"
          fi
          echo "PACKAGE_FILE: $PACKAGE_FILE"
          PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"${PACKAGE_FILE}
          echo "PACKAGE_PATH: $PACKAGE_PATH"
          wget ${PACKAGE_PATH} -P ${PACKAGE_DOWNLOAD_DIR}
          pushd ${PACKAGE_EXTRACT_DIR}
          rpm2cpio ../${PACKAGE_DOWNLOAD_DIR}/$PACKAGE_FILE | cpio -div
          popd
        fi
      done
    elif [ ${OS_TYPE} == "anolis" ]; then
      for (( i =0; i < ${n}; i++))
      do
        PACKAGE_DIST="Anolis"
        PACKAGE_DIST_VER=${OS_VERSION}
        if [ ${arr_modules[$i]} == "cntoolkit" ]; then
          PACKAGE_FILE=${arr_modules[$i]}"-"${arr_vers[$i]}".an8.${PACKAGE_ARCH}.rpm"
          echo "PACKAGE_FILE: $PACKAGE_FILE"
          PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"${PACKAGE_FILE}
          echo "PACKAGE_PATH: $PACKAGE_PATH"
          MODULE_PURE_VER=`echo ${arr_vers[$i]} | cut -d '-' -f 1`
          wget ${PACKAGE_PATH} -P ${PACKAGE_DOWNLOAD_DIR} -nv
          # extract in extract dir
          pushd ${PACKAGE_DOWNLOAD_DIR}
          rpm2cpio $PACKAGE_FILE | cpio -div
          popd
          pushd ${PACKAGE_EXTRACT_DIR}
          for filename in ../${PACKAGE_DOWNLOAD_DIR}/var/${arr_modules[$i]}"-"$MODULE_PURE_VER/*.rpm; do
            rpm2cpio $filename | cpio -div
          done
          popd
        else
          if [[ ${arr_branch[$i]} == "daily" || ${arr_branch[$i]} == "temp" ]];then
            VERSION_FILE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/version.txt"
            wget $VERSION_FILE_PATH -P ${PACKAGE_DOWNLOAD_DIR} -nv
            PACKAGE_FILE=$(cat ${PACKAGE_DOWNLOAD_DIR}"/version.txt" | awk -F '=' '{if($1=="file") print $2}')
            rm ${PACKAGE_DOWNLOAD_DIR}"/version.txt"
          else
            PACKAGE_FILE=${arr_modules[$i]}"-"${arr_vers[$i]}".an8.${PACKAGE_ARCH}.rpm"
          fi
          echo "PACKAGE_FILE: $PACKAGE_FILE"
          PACKAGE_PATH=${PACKAGE_SERVER}"/"${arr_branch[$i]}"/"${arr_modules[$i]}"/"${PACKAGE_OS}"/"${PACKAGE_ARCH}"/"${PACKAGE_DIST}"/"${PACKAGE_DIST_VER}"/"${arr_vers[$i]}"/"${PACKAGE_FILE}
          echo "PACKAGE_PATH: $PACKAGE_PATH"
          wget ${PACKAGE_PATH} -P ${PACKAGE_DOWNLOAD_DIR} -nv
          pushd ${PACKAGE_EXTRACT_DIR}
          rpm2cpio ../${PACKAGE_DOWNLOAD_DIR}/$PACKAGE_FILE | cpio -div
          popd
        fi
      done
    fi
  fi


  cp -r ${PACKAGE_EXTRACT_DIR}/usr/local/neuware/* ${NEUWARE_HOME}
  rm -rf ${PACKAGE_DOWNLOAD_DIR}
  rm -rf ${PACKAGE_EXTRACT_DIR}
  echo "****************************************************************"
  echo "* PLEASE CONFIGURE NEUWARE_HOME"
  echo "* export NEUWARE_HOME=${PWD}/${NEUWARE_HOME}"
  echo "* export LD_LIBRARY_PATH=${PWD}/${NEUWARE_HOME}/lib64:${PWD}/${NEUWARE_HOME}/lib"
  echo "* DONE"
  echo "****************************************************************"
}

# build wheel in docker
build_wheel_func(){
  set -x
  container_name=$(docker run \
    -e PYTHON_VERSION=${PYTHON_VERSION} \
    -e PYTORCH_VERSION=${PYTORCH_VERSION} \
    -e TORCH_MLU_BRANCH=${TORCH_MLU_BRANCH} \
    -e TORCHAUDIO_MLU_BRANCH=${TORCHAUDIO_MLU_BRANCH} \
    -e TORCH_MLU_COMMIT_ID=${TORCH_MLU_COMMIT_ID} \
    -e VISION_VERSION=${VISION_VERSION} \
    -e AUDIO_VERSION=${AUDIO_VERSION} \
    --tty \
    --detach \
    --net=host \
    -v ${ABS_DIR_PATH}:/cambricon \
    -w / \
    ${BUILDER_DOCKER}
  )
  docker cp ${ABS_DIR_PATH}/build_wheel.sh ${container_name}:/
  docker exec -t ${container_name} bash -c "bash /build_wheel.sh"
  docker cp ${container_name}:/wheelhouse ./wheel_${PY_SUFFIX}
  docker rm -f ${container_name}
  set +x
}

# install wheel in docker
install_docker_func(){
  install_docker_cmd="docker build --no-cache --pull --network=host --rm          \
                      --build-arg pytorch_version=${PYTORCH_VERSION}              \
                      --build-arg torch_mlu_branch=${TORCH_MLU_BRANCH}            \
                      --build-arg torchaudio_mlu_branch=${TORCHAUDIO_MLU_BRANCH}  \
                      --build-arg torch_mlu_commit_id=${TORCH_MLU_COMMIT_ID}      \
                      --build-arg vision_version=${VISION_VERSION}                \
                      --build-arg audio_version=${AUDIO_VERSION}                  \
                      --build-arg pytorch_models_branch=${PYTORCH_MODELS_BRANCH}  \
                      --build-arg benchmark_version=${BENCHMARK_VERSION}          \
                      --build-arg driver_version=${DRIVER_VERSION}                \
                      --build-arg cntoolkit_version=${CNTOOLKIT_VERSION}          \
                      --build-arg cnnl_version=${CNNL_VERSION}                    \
                      --build-arg cnnlextra_version=${CNNLExtra_VERSION}          \
                      --build-arg cncl_version=${CNCL_VERSION}                    \
                      --build-arg cncv_version=${CNCV_VERSION}                    \
                      --build-arg cndali_version=${CNDALI_VERSION}                \
                      --build-arg python_version=${PYTHON_VERSION}                \
                      --build-arg py_suffix=${PY_SUFFIX}                          \
                      --build-arg genesis_version=${GENESIS_VERSION}              \
                      -t ${IMAGE_DOCKER_NAME}:${TAG} -f ${DOCKER_FILE} ."
  echo "install_docker_func command: "$install_docker_cmd
  eval $install_docker_cmd
}

# pack src func in host
pack_src_func(){
	PYTORCH_PACKAGE="cambricon_pytorch"
	rm -rf ${PYTORCH_PACKAGE}
	mkdir ${PYTORCH_PACKAGE}
	pushd ${PYTORCH_PACKAGE}

	# create directory for package
	PYTORCH_SRC_PACKAGE="pytorch/src"
	if [ ! -d ${PYTORCH_SRC_PACKAGE} ]; then
	  mkdir -p ${PYTORCH_SRC_PACKAGE}
	else
	  rm -rf ${PYTORCH_SRC_PACKAGE}/*
	fi

  # step 1: git clone source codes
  pushd ${PYTORCH_SRC_PACKAGE}
  git clone http://gitlab.software.cambricon.com/neuware/oss/pytorch/torch_mlu.git -b $TORCH_MLU_BRANCH --single-branch
  pushd torch_mlu
  git checkout "${TORCH_MLU_COMMIT_ID}"
  popd
  git clone http://gitlab.software.cambricon.com/neuware/oss/pytorch/torchaudio_mlu.git -b $TORCHAUDIO_MLU_BRANCH --single-branch
  git clone http://gitlab.software.cambricon.com/neuware/pytorch_models.git -b $PYTORCH_MODELS_BRANCH --depth 1
  bash $TORCH_MLU_PATH/.jenkins/pipeline/enable_git_url_cache.sh
  pytorch_commit_id=`cat $TORCH_MLU_PATH/pytorch_patches/commit_id | grep pytorch | awk -F ":" '{print$2}'`
  if [[ "$PYTORCH_VERSION" == "main" ]] || [[ "$PYTORCH_VERSION" == "release"* ]]; then
      git clone --recursive https://github.com/pytorch/pytorch.git -b $PYTORCH_VERSION
      pushd ./pytorch
      git checkout $pytorch_commit_id
      git submodule update --recursive
      if [ $? -eq 0 ]; then
          echo "submodule updated successfully."
      else
          echo "submodule update failed."
          exit 1
      fi
      popd
  else
      git clone --recursive https://github.com/pytorch/pytorch.git -b "v${PYTORCH_VERSION}"
  fi
  popd

  # step 2: copy shell script to cambricon_pytorch dir
  cp ${PYTORCH_SRC_PACKAGE}/torch_mlu/scripts/release/env_pytorch.sh .

  # step 3: remove docs in torch_mlu and pytorch_models/Training/tools/internal.
  rm -rf ${PYTORCH_SRC_PACKAGE}/torch_mlu/docs
  rm -rf ${PYTORCH_SRC_PACKAGE}/pytorch_models/Training/tools/internal

  # step 4: remove git/jenkins/other info
  pushd ${PYTORCH_SRC_PACKAGE}
  ./torch_mlu/scripts/release/torch_mlu_trim_files.sh
  popd

	popd # to cambricon_pytorch/../

  # step 5: pack
  pack_src_cmd="tar cfz Cambricon-PyTorch-$TAG.tar.gz ${PYTORCH_PACKAGE}"
  eval $pack_src_cmd

  # step 6: remove
  rm -rf $PYTORCH_PACKAGE
}

if [ $RELEASE_TYPE == "wheel" ]; then
  echo "=== BUILD WHEEL ==="
  build_wheel_func
elif [ $RELEASE_TYPE == "docker" ]; then
  echo "=== BUILD DOCKER ==="
  read_ver_func
  fetch_neuware_ver_func
  install_docker_func
elif [ ${RELEASE_TYPE} == "src" ]; then
  echo "=== RELEASE SRC ==="
  pack_src_func
elif [ ${RELEASE_TYPE} == "dep" ]; then
  echo "=== DOWNLOAD DEP ==="
  read_ver_func
  fetch_cn_dep_func
else
  echo "unrecognized RELEASE_TYPE: "$RELEASE_TYPE
fi
