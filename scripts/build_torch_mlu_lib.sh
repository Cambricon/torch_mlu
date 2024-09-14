#!/usr/bin/env bash
#! /bin/bash
# Shell script used to build the c++ extension lib

### configure C++ compiler
export compiler=$(which g++)
export compiler_CXX=$(which $CXX)

### get default g++ version
MAJOR=$(echo __GNUC__ | $compiler -E -xc - | tail -n 1)
MINOR=$(echo __GNUC_MINOR__ | $compiler -E -xc - | tail -n 1)
PATCHLEVEL=$(echo __GNUC_PATCHLEVEL__ | $compiler -E -xc - | tail -n 1)

### check whether the CXX environment variable has been set or not.
### if set, get the g++ version which is pointed by CXX environment variable.
CXX_MAJOR=''
CXX_MINOR=''
CXX_PATCHLEVEL=''
if [ -n "$compiler_CXX" ];then
    CXX_MAJOR=$(echo __GNUC__ | $compiler_CXX -E -xc - | tail -n 1)
    CXX_MINOR=$(echo __GNUC_MINOR__ | $compiler_CXX -E -xc - | tail -n 1)
    CXX_PATCHLEVEL=$(echo __GNUC_PATCHLEVEL__ | $compiler_CXX -E -xc - | tail -n 1)
fi

centos_file="/etc/redhat-release"

### Check the g++ version.
if (([[ $MAJOR < "7" ]] && [[ $CXX_MAJOR < "7" ]]) || ([[ $MAJOR > "9" ]] && [[ $CXX_MAJOR > "9" ]])) && [ $USE_MAGICMIND == "ON" ];then
    echo -e "\033[33mWhen enabling USE_MAGICMIND env, the GCC version should be in gcc-7, gcc-8, gcc-9 and g++-7, g++-8, g++-9, USE_MAGICMIND env is enabled in default\033[0m"
    echo -e "\033[33mCurrent version is gcc-$MAJOR.$MINOR.$PATCHLEVEL \033[0m"
    if [ -n "$compiler_CXX" ];then
        echo -e "\033[33mCurrent CXX environment variable version is gcc-$CXX_MAJOR.$CXX_MINOR.$CXX_PATCHLEVEL \033[0m"
    fi
    echo -e "\033[33mPlease install the suitable GCC version on your system and try again \033[0m"

    exit
elif (([[ $MAJOR < "7" ]] && [[ $CXX_MAJOR < "7" ]]) || ([[ $MAJOR > "9" ]] && [[ $CXX_MAJOR > "9" ]])) &&  [ $USE_MAGICMIND == "OFF" ]; then
    echo -e "\033[33mWhen disabling USE_MAGICMIND env, the recommended GCC version is gcc-7, gcc-8, gcc-9 and g++-7, g++-8, g++-9, USE_MAGICMIND env is enabled in default\033[0m"
    echo -e "\033[33mCurrent version is gcc-$MAJOR.$MINOR.$PATCHLEVEL \033[0m"
    if [ -n "$compiler_CXX" ];then
        echo -e "\033[33mCurrent CXX environment variable is gcc-$CXX_MAJOR.$CXX_MINOR.$CXX_PATCHLEVEL \033[0m"
    fi
    echo -e "\033[33mPlease install the suitable GCC version on your system and try again \033[0m"
fi

set -e
if [ -z "$MAX_JOBS" ]; then
    MAX_JOBS="$(getconf _NPROCESSORS_ONLN)"
fi

# Set options
RERUN_CMAKE=1

# Set PATH
export PATH="${NEUWARE_HOME}/bin":$PATH

# Set Cmake command
CMAKE_COMMAND="cmake"
# We test the presence of cmake3 (for platforms like CentOS and Ubuntu 14.04)
# and use the newer of cmake and cmake3 if so.
if [[ -x "$(command -v cmake3)" ]]; then
    if [[ -x "$(command -v cmake)" ]]; then
        # have both cmake and cmake3, compare versions
        # Usually cmake --version returns two lines,
        #   cmake version #.##.##
        #   <an empty line>
        # On the nightly machines it returns one line
        #   cmake3 version 3.11.0 CMake suite maintained and supported by Kitware (kitware.com/cmake).
        # Thus we extract the line that has 'version' in it and hope the actual
        # version number is gonna be the 3rd element
        CMAKE_VERSION=$(cmake --version | grep 'version' | awk '{print $3}' | awk -F. '{print $1"."$2"."$3}')
        CMAKE3_VERSION=$(cmake3 --version | grep 'version' | awk '{print $3}' | awk -F. '{print $1"."$2"."$3}')
        CMAKE3_NEEDED=$($PYTHON_EXECUTABLE -c "from distutils.version import StrictVersion; print(1 if StrictVersion(\"${CMAKE_VERSION}\")     < StrictVersion(\"3.5.0\") and StrictVersion(\"${CMAKE3_VERSION}\") > StrictVersion(\"${CMAKE_VERSION}\") else 0)")
    else
        # don't have cmake
        CMAKE3_NEEDED=1
    fi
    if [[ $CMAKE3_NEEDED == "1" ]]; then
        CMAKE_COMMAND="cmake3"
    fi
    unset CMAKE_VERSION CMAKE3_VERSION CMAKE3_NEEDED
fi

CMAKE_INSTALL=${CMAKE_INSTALL-make install}

BASE_DIR=$(cd $(dirname "$0")/.. && printf "%q\n" "$(pwd)")
TORCH_MLU_CSRC_DIR="$BASE_DIR/torch_mlu/csrc"
TORCH_MLU_INSTALL_DIR="$TORCH_MLU_CSRC_DIR"

# Set compile option
C_FLAGS=""
CXX_FLAGS=$EXTRA_COMPILE_ARGS

## Used to check if certain env variables are enabled
function check_env_flag () {
    local env_name=$1
    local env_true_values=("ON" "1" "YES" "TRUE" "Y")
    local out_flag=""
    for true_value in ${env_true_values[@]};
    do
        if [[ ${!env_name^^} == ${true_value} ]];then
            out_flag=${!env_name^^}
            break
        fi
    done
    echo $out_flag
}

## Configure cmake build type according to envs, with priority: CMAKE_BUILD_TYPE > DEBUG > RelWithDebInfo
cmake_build_type="${CMAKE_BUILD_TYPE}"
if [[ -z ${cmake_build_type} ]]; then

    debug=$(check_env_flag "DEBUG")
    rel_with_deb=$(check_env_flag "REL_WITH_DEB_INFO")

    if [[ -n ${debug} ]];then
        cmake_build_type="DEBUG"
    elif [[ -n ${rel_with_deb} ]];then
        cmake_build_type="RelWithDebInfo"
    else
        cmake_build_type="Release"
    fi
fi

function build_ext_lib() {
    if [[ $RERUN_CMAKE -eq 1 ]] || [ ! -f CMakeCache.txt ]; then
        ${CMAKE_COMMAND} $TORCH_MLU_CSRC_DIR \
                         -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH \
                         -DNEUWARE_HOME=$NEUWARE_HOME\
                         -DPYTORCH_WHEEL_DIR=$PYTORCH_WHEEL_DIR \
                         -DCMAKE_C_FLAGS="$C_FLAGS" \
                         -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
                         -DCMAKE_INSTALL_PREFIX="$TORCH_MLU_INSTALL_DIR" \
                         -DCMAKE_BUILD_TYPE=$cmake_build_type \
                         -DBUILD_TEST="$BUILD_TEST" \
                         -DBUILD_LIBTORCH="$BUILD_LIBTORCH" \
                         -DUSE_PYTHON="$USE_PYTHON" \
                         -DUSE_BANG="$USE_BANG" \
                         -DUSE_MLUOP="$USE_MLUOP" \
                         -DUSE_CNCL="$USE_CNCL" \
                         -DUSE_PROFILE="$USE_PROFILE" \
                         -DUSE_MAGICMIND="$USE_MAGICMIND" \
                         -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
                         -DPYTHON_LIBRARY="$PYTHON_LIBRARY" \
                         -DPYTHON_INCLUDE_DIR="$PYTHON_INCLUDE_DIR" \
                         -DGLIBCXX_USE_CXX11_ABI="$GLIBCXX_USE_CXX11_ABI"
    fi

    ${CMAKE_INSTALL} -j"$MAX_JOBS"
}

# In the torch/lib directory, create an installation directory
mkdir -p $TORCH_MLU_INSTALL_DIR

#Build
build_ext_lib
