#!/bin/bash
set -e


OS_VERSION=0
WORK_PATH=$(cd $(dirname $0);pwd)

function usage()
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  optional params:"
    echo "|                 -o choice os, 0)centos7, 1)ubuntu, defaultly be 0."
    echo "|                 -p work path, eg. YOUR_PATH_TO/CambriconNeuware_v2.0.1_CentOS7, defaultly be current path."
    echo "|  eg. ./gen_cambricon_neuware_lib.sh -o 0 -p YOUR_PATH_TO/CambriconNeuware_v2.0.1_CentOS7"
    echo "|     which means gen neuware lib for centos7."
    echo "|  note: root privilege is required."
    echo "-------------------------------------------------------------"
}

while getopts "o:p:" opt; do
case "$opt" in
    o) OS_VERSION=$OPTARG;;
    p) WORK_PATH=$OPTARG;;
    ?) echo "there is unrecognized optional parameter."; usage; exit 1;;
esac
done

function gen_cntoolkit() {
  if [ $OS_VERSION -eq 0 ];then
    bool_enabled_epel=$(yum repolist | grep epel)
    if [ "$bool_enabled_epel" == "" ];then sudo yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
      sudo yum update
      sudo yum install centos-release-scl
      sudo yum install devtoolset-7
      sudo scl enable devtoolset-7 bash
    fi
    cntoolkit_path=$WORK_PATH/cntoolkit/
    pushd $cntoolkit_path
      cntoolkit_rpm=$(ls *.rpm)
      sudo rpm --install  "$cntoolkit_rpm"
      sudo yum clean expire-cache
      sudo yum  -y install cntoolkit-cloud cntoolkit-edge
    popd
  else
    cntoolkit_path=$WORK_PATH/cntoolkit/
    pushd $cntoolkit_path
      cntoolkit_deb=$(ls *.deb)
      echo $cntoolkit_deb
      sudo dpkg -i $cntoolkit_deb
      sudo apt update
      sudo apt-get install cntoolkit-cloud cntoolkit-edge
    popd
  fi
}

function gen_cncl() {
  cncl_path=$WORK_PATH/cncl/
  pushd $cncl_path
    if [ $OS_VERSION -eq 0 ];then
      cncl_rpm=$(ls *.rpm)
      sudo rpm -ivh $cncl_rpm
    else
      cncl_deb=$(ls *.deb)
      sudo dpkg -i $cncl_deb
    fi
  popd
}

function gen_cnnl(){
  cnnl_path=$WORK_PATH/cnnl/
  pushd $cnnl_path
    if [ $OS_VERSION -eq 0 ];then
      cnnl_rpm=$(ls *.rpm | awk '{if($1!~"static") print $1}')
      sudo yum -y install $cnnl_rpm
    else
      cnnl_deb=$(ls *.deb | awk '{if($1!~"static") print $1}')
      sudo dpkg -i  $cnnl_deb
    fi
  popd
}

function gen_magicmind() {
  magicmind_path=$WORK_PATH/magicmind/
  pushd $magicmind_path
  if [ $OS_VERSION -eq 0 ];then
    magicmind_rpm=$(ls *.rpm)
    sudo rpm -i $magicmind_rpm
  else
    magicmind_deb=$(ls *.deb)
    sudo dpkg -i $magicmind_deb
  fi
  popd
}

gen_cntoolkit
gen_cncl
gen_cnnl
gen_magicmind
