#!/bin/bash
set -e

# Get the path of current script
CUR_DIR=$(cd $(dirname $0);pwd)
# Get the root path of torch_mlu
TORCH_MLU_HOME=$CUR_DIR/../

# NEUWARE_HOME Env should be set before building torch_mlu
if [ -z ${NEUWARE_HOME} ]; then
    echo "Error: please set environment variable NEUWARE_HOME, export NEUWARE_HOME=your Cambricon neuware package dir!"
    exit 1
fi

install_pytorch() {
    pushd ${TORCH_MLU_HOME}

    version_type=`cat ./scripts/version.info | grep version_type | awk -F ":" '{print $2}'`
    pytorch=`cat ./scripts/version.info | grep pytorch | awk -F ":" '{print $2}'`
    vision=`cat ./scripts/version.info | grep vision | awk -F ":" '{print $2}'`
    audio=`cat ./scripts/version.info | grep audio | awk -F ":" '{print $2}'`
    pip uninstall -y torch torchvision torchaudio
    if [ "$version_type" == "main" ]; then
        echo "Install preview version of PyTorch ..."
        pip install torch==$pytorch --index-url http://mirrors.cambricon.com/pytorch/whl/nightly/ --trusted-host mirrors.cambricon.com
        pip install torchvision==$vision --index-url http://mirrors.cambricon.com/pytorch/whl/nightly/ --trusted-host mirrors.cambricon.com
        pip install torchaudio==$audio --index-url http://mirrors.cambricon.com/pytorch/whl/nightly/ --trusted-host mirrors.cambricon.com
    elif [ "$version_type" == "release" ]; then
        echo "Install release version of PyTorch ..."
        pip install torch==$pytorch --index-url http://mirrors.cambricon.com/pytorch/whl/ --trusted-host mirrors.cambricon.com
        pip install torchvision==$vision --index-url http://mirrors.cambricon.com/pytorch/whl/ --trusted-host mirrors.cambricon.com
        pip install torchaudio==$audio --index-url http://mirrors.cambricon.com/pytorch/whl/ --trusted-host mirrors.cambricon.com
    elif [ "$version_type" == "rc" ]; then
        echo "Install release candidate version of PyTorch ..."
        python_version=`python -V 2>&1|awk '{print $2}'|awk -F '.' '{print $1$2}'`
        torch_whl="torch-${pytorch}-cp${python_version}-cp${python_version}-linux_x86_64.whl"
        vision_whl="torchvision-${vision}-cp${python_version}-cp${python_version}-linux_x86_64.whl"
        audio_whl="torchaudio-${audio}-cp${python_version}-cp${python_version}-linux_x86_64.whl"
        wget -nv "http://mirrors.cambricon.com/pytorch/whl/test/torch/${torch_whl}"
        wget -nv "http://mirrors.cambricon.com/pytorch/whl/test/torchvision/${vision_whl}"
        wget -nv "http://mirrors.cambricon.com/pytorch/whl/test/torchaudio/${audio_whl}"
        pip install --force-reinstall ${torch_whl} ${vision_whl} ${audio_whl}
        rm ${torch_whl} ${vision_whl} ${audio_whl}
    else
        echo "Unknown install type: $version_type"
        exit 1
    fi

    popd
}

build_install_torch_mlu() {
    pushd $TORCH_MLU_HOME
    if [ -d ".git" ];then
        python setup.py clean
    else
        rm -rf build torch_mlu/csrc/lib
    fi

    echo "====================install requirements==============================="
    pip install -r requirements.txt --timeout=60 --retries=30 #default timeout=15 retries=5

    echo "=========================build torch_mlu==================================="
    python setup.py install

    popd
}

main() {
    install_pytorch
    build_install_torch_mlu
}

main
