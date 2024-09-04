#!/usr/bin/env bash
set -ex

function checkEnv() {
    if [[ -z "${PYTHON_VERSION}" ]]; then
        echo "ERROR : PYTHON_VERSION is not set"
        exit 1
    fi
    if [[ -z "${PYTORCH_VERSION}" ]]; then
        echo "ERROR : PYTORCH_VERSION is not set"
        exit 1
    fi
    if [[ -z "${TORCH_MLU_BRANCH}" ]]; then
        echo "ERROR : TORCH_MLU_BRANCH is not set"
        exit 1
    fi
    if [[ -z "${TORCH_MLU_COMMIT_ID}" ]]; then
        echo "ERROR : TORCH_MLU_COMMIT_ID is not set"
        exit 1
    fi
    if [[ -z "${VISION_VERSION}" ]]; then
        echo "ERROR : VISION_VERSION is not set"
        exit 1
    fi
    if [[ -z "${AUDIO_VERSION}" ]]; then
        echo "ERROR : AUDIO_VERSION is not set"
        exit 1
    fi
}

checkEnv

########################################################
# Prepare env
#######################################################
git clone http://gitlab.software.cambricon.com/neuware/oss/pytorch/torch_mlu.git -b "${TORCH_MLU_BRANCH}" --single-branch
pushd torch_mlu
git checkout "${TORCH_MLU_COMMIT_ID}"
popd

TORCH_MLU_HOME="/torch_mlu"

bash torch_mlu/scripts/release/independent_build.sh -r dep -o centos -v 7
export NEUWARE_HOME="/neuware_home"

WHEELHOUSE_DIR="wheelhouse"

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

retry yum install -q -y zip openssl

export CMAKE_LIBRARY_PATH="/opt/intel/lib:/lib:$CMAKE_LIBRARY_PATH"
export CMAKE_INCLUDE_PATH="/opt/intel/include:$CMAKE_INCLUDE_PATH"

if [[ -e /opt/openssl ]]; then
    export OPENSSL_ROOT_DIR=/opt/openssl
    export CMAKE_INCLUDE_PATH="/opt/openssl/include":$CMAKE_INCLUDE_PATH
fi

python_nodot="$(echo $PYTHON_VERSION | tr -d .)"
PYTHON_VERSION="cp${python_nodot}-cp${python_nodot}"
if [[ ${python_nodot} -ge 310 ]]; then
    py_majmin="${PYTHON_VERSION:2:1}.${PYTHON_VERSION:3:2}"
else
    py_majmin="${PYTHON_VERSION:2:1}.${PYTHON_VERSION:3:1}"
fi

pydir="/opt/python/$PYTHON_VERSION"
export PATH="$pydir/bin:$PATH"

if compgen -G "${TORCH_MLU_HOME}/pytorch_patches/*diff" > /dev/null; then
    echo "There are patches to apply, clone pytorch source..."
    bash ${TORCH_MLU_HOME}/.jenkins/pipeline/enable_git_url_cache.sh
    export PYTORCH_HOME="/pytorch"
    if [[ "$PYTORCH_VERSION" == "main" ]] || [[ "$PYTORCH_VERSION" == "release"* ]]; then
        git clone https://github.com/pytorch/pytorch.git --single-branch -b $PYTORCH_VERSION
        pytorch_commit_id=`cat ./torch_mlu/pytorch_patches/commit_id | grep pytorch | awk -F ":" '{print$2}'`
        if [[ -n "${pytorch_commit_id}" ]]; then
            pushd "$PYTORCH_HOME"
            git checkout "${pytorch_commit_id}"
            git submodule sync
            git submodule update --init --recursive
            popd
        fi
        git clone --recursive https://github.com/pytorch/vision.git --single-branch -b $VISION_VERSION
        vision_commit_id=`cat ./torch_mlu/pytorch_patches/commit_id | grep vision | awk -F ":" '{print$2}'`
        pushd "/vision"
        git checkout "${vision_commit_id:=$(cat /pytorch/.github/ci_commit_pins/vision.txt)}"
        git submodule sync
        git submodule update --init --recursive
        popd
        git clone --recursive https://github.com/pytorch/audio.git --single-branch -b $AUDIO_VERSION
        audio_commit_id=`cat ./torch_mlu/pytorch_patches/commit_id | grep audio | awk -F ":" '{print$2}'`
        pushd "/audio"
        git checkout "${audio_commit_id:=$(cat /pytorch/.github/ci_commit_pins/audio.txt)}"
        git submodule sync
        git submodule update --init --recursive
        popd
    else
        git clone --recursive https://github.com/pytorch/pytorch.git --single-branch -b "v${PYTORCH_VERSION}"
        pushd "pytorch"
        git submodule sync
        git submodule update --init --recursive
        popd
        export PYTORCH_BUILD_VERSION="${PYTORCH_VERSION}+cpu"
        export PYTORCH_BUILD_NUMBER=1
    fi

    export TH_BINARY_BUILD=1
    export USE_CUDA=0
    export USE_GOLD_LINKER="ON"
    export USE_GLOO_WITH_OPENSSL="ON"

    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
    DEPS_LIST=("$LIBGOMP_PATH")
    DEPS_SONAME=("libgomp.so.1")

    mkdir -p /tmp/$WHEELHOUSE_DIR
    export PATCHELF_BIN=/usr/local/bin/patchelf
    patchelf_version=$($PATCHELF_BIN --version)
    echo "patchelf version: " $patchelf_version
    if [[ "$patchelf_version" == "patchelf 0.9" ]]; then
        echo "Your patchelf version is too old. Please use version >= 0.10."
        exit 1
    fi

    pushd "$TORCH_MLU_HOME/scripts"
    ./apply_patches_to_pytorch.sh
    popd

    ########################################################
    # Compile torch wheels
    #######################################################
    pushd "$PYTORCH_HOME"
    python setup.py clean
    retry pip install -qr requirements.txt
    case ${PYTHON_VERSION} in
        cp38*)
            retry pip install -q numpy==1.15
            ;;
        cp31*)
            retry pip install -q numpy==2.0.0
            ;;
        # Should catch 3.9+
        *)
            retry pip install -q numpy==2.0.0
            ;;
    esac

    export _GLIBCXX_USE_CXX11_ABI=0

    echo "Calling setup.py bdist at $(date)"
    BUILD_LIBTORCH_CPU_WITH_DEBUG=0 python setup.py bdist_wheel -d /tmp/$WHEELHOUSE_DIR
    echo "Finished setup.py bdist at $(date)"
    popd  # PYTORCH_HOME

    #######################################################################
    # ADD DEPENDENCIES INTO THE TORCH WHEEL
    #
    # auditwheel repair doesn't work correctly and is buggy
    # so manually do the work of copying dependency libs and patchelfing
    # and fixing RECORDS entries correctly
    ######################################################################

    fname_with_sha256() {
        HASH=$(sha256sum $1 | cut -c1-8)
        DIRNAME=$(dirname $1)
        BASENAME=$(basename $1)
        INITNAME=$(echo $BASENAME | cut -f1 -d".")
        ENDNAME=$(echo $BASENAME | cut -f 2- -d".")
        echo "$DIRNAME/$INITNAME-$HASH.$ENDNAME"
    }

    make_wheel_record() {
        FPATH=$1
        if echo $FPATH | grep RECORD >/dev/null 2>&1; then
            # if the RECORD file, then
            echo "$FPATH,,"
        else
            HASH=$(openssl dgst -sha256 -binary $FPATH | openssl base64 | sed -e 's/+/-/g' | sed -e 's/\//_/g' | sed -e 's/=//g')
            FSIZE=$(ls -nl $FPATH | awk '{print $5}')
            echo "$FPATH,sha256=$HASH,$FSIZE"
        fi
    }

    replace_needed_sofiles() {
        find $1 -name '*.so*' | while read sofile; do
            origname=$2
            patchedname=$3
            if [[ "$origname" != "$patchedname" ]]; then
                set +e
                origname=$($PATCHELF_BIN --print-needed $sofile | grep "$origname.*")
                ERRCODE=$?
                set -e
                if [ "$ERRCODE" -eq "0" ]; then
                    echo "patching $sofile entry $origname to $patchedname"
                    $PATCHELF_BIN --replace-needed $origname $patchedname $sofile
                fi
            fi
        done
    }

    echo 'Built this wheel:'
    ls /tmp/$WHEELHOUSE_DIR
    mkdir -p "/$WHEELHOUSE_DIR"
    mv /tmp/$WHEELHOUSE_DIR/torch*linux*.whl /$WHEELHOUSE_DIR/
    rm -rf /tmp/$WHEELHOUSE_DIR
    rm -rf /tmp_dir
    mkdir /tmp_dir
    pushd /tmp_dir
    for pkg in /$WHEELHOUSE_DIR/torch*linux*.whl; do

        # if the glob didn't match anything
        if [[ ! -e $pkg ]]; then
            continue
        fi

        rm -rf tmp
        mkdir -p tmp
        cd tmp
        cp $pkg .

        unzip -q $(basename $pkg)
        rm -f $(basename $pkg)

        PREFIX=torch

        # copy over needed dependent .so files over and tag them with their hash
        patched=()
        for filepath in "${DEPS_LIST[@]}"; do
            filename=$(basename $filepath)
            destpath=$PREFIX/lib/$filename
            if [[ "$filepath" != "$destpath" ]]; then
                cp $filepath $destpath
            fi

            patchedpath=$(fname_with_sha256 $destpath)
            patchedname=$(basename $patchedpath)
            if [[ "$destpath" != "$patchedpath" ]]; then
                mv $destpath $patchedpath
            fi
            patched+=("$patchedname")
            echo "Copied $filepath to $patchedpath"
        done

        echo "patching to fix the so names to the hashed names"
        for ((i=0;i<${#DEPS_LIST[@]};++i)); do
            replace_needed_sofiles $PREFIX ${DEPS_SONAME[i]} ${patched[i]}
            # do the same for caffe2, if it exists
            if [[ -d caffe2 ]]; then
                replace_needed_sofiles caffe2 ${DEPS_SONAME[i]} ${patched[i]}
            fi
        done

        # set RPATH of _C.so and similar to $ORIGIN, $ORIGIN/lib
        find $PREFIX -maxdepth 1 -type f -name "*.so*" | while read sofile; do
            echo "Setting rpath of $sofile to ${C_SO_RPATH:-'$ORIGIN:$ORIGIN/lib'}"
            $PATCHELF_BIN --set-rpath ${C_SO_RPATH:-'$ORIGIN:$ORIGIN/lib'} ${FORCE_RPATH:-} $sofile
            $PATCHELF_BIN --print-rpath $sofile
        done

        # set RPATH of lib/ files to $ORIGIN
        find $PREFIX/lib -maxdepth 1 -type f -name "*.so*" | while read sofile; do
            echo "Setting rpath of $sofile to ${LIB_SO_RPATH:-'$ORIGIN'}"
            $PATCHELF_BIN --set-rpath ${LIB_SO_RPATH:-'$ORIGIN'} ${FORCE_RPATH:-} $sofile
            $PATCHELF_BIN --print-rpath $sofile
        done

        # regenerate the RECORD file with new hashes
        record_file=$(echo $(basename $pkg) | sed -e 's/-cp.*$/.dist-info\/RECORD/g')
        if [[ -e $record_file ]]; then
            echo "Generating new record file $record_file"
            : > "$record_file"
            # generate records for folders in wheel
            find * -type f | while read fname; do
                make_wheel_record "$fname" >>"$record_file"
            done
        fi

        # zip up the wheel back
        zip -rq $(basename $pkg) $PREIX*

        # replace original wheel
        rm -f $pkg
        mv $(basename $pkg) $pkg
        cd ..
        rm -rf tmp
    done
    popd  # /tmp_dir
else
    # download torch/vision/audio whl.
    version_type=`cat ${TORCH_MLU_HOME}/scripts/version.info | grep version_type | awk -F ":" '{print $2}'`
    pytorch=`cat ${TORCH_MLU_HOME}/scripts/version.info | grep pytorch | awk -F ":" '{print $2}'`
    vision=`cat ${TORCH_MLU_HOME}/scripts/version.info | grep vision | awk -F ":" '{print $2}'`
    audio=`cat ${TORCH_MLU_HOME}/scripts/version.info | grep audio | awk -F ":" '{print $2}'`
    if [ "$version_type" == "main" ]; then
        pip download torch==$pytorch --index-url http://mirrors.cambricon.com/pytorch/whl/nightly/ --trusted-host mirrors.cambricon.com -d /$WHEELHOUSE_DIR/ --no-deps
        pip download torchvision==$vision --index-url http://mirrors.cambricon.com/pytorch/whl/nightly/ --trusted-host mirrors.cambricon.com -d /$WHEELHOUSE_DIR/ --no-deps
        pip download torchaudio==$audio --index-url http://mirrors.cambricon.com/pytorch/whl/nightly/ --trusted-host mirrors.cambricon.com -d /$WHEELHOUSE_DIR/ --no-deps
    elif [ "$version_type" == "release" ]; then
        pip download torch==$pytorch --index-url http://mirrors.cambricon.com/pytorch/whl/ --trusted-host mirrors.cambricon.com -d /$WHEELHOUSE_DIR/ --no-deps
        pip download torchvision==$vision --index-url http://mirrors.cambricon.com/pytorch/whl/ --trusted-host mirrors.cambricon.com -d /$WHEELHOUSE_DIR/ --no-deps
        pip download torchaudio==$audio --index-url http://mirrors.cambricon.com/pytorch/whl/ --trusted-host mirrors.cambricon.com -d /$WHEELHOUSE_DIR/ --no-deps
    elif [ "$version_type" == "rc" ]; then
        torch_whl="torch-${pytorch}-cp${python_nodot}-cp${python_nodot}-linux_x86_64.whl"
        vision_whl="torchvision-${vision}-cp${python_nodot}-cp${python_nodot}-linux_x86_64.whl"
        audio_whl="torchaudio-${audio}-cp${python_nodot}-cp${python_nodot}-linux_x86_64.whl"
        wget -nv -P /$WHEELHOUSE_DIR "http://mirrors.cambricon.com/pytorch/whl/test/torch/${torch_whl}"
        wget -nv -P /$WHEELHOUSE_DIR "http://mirrors.cambricon.com/pytorch/whl/test/torchvision/${vision_whl}"
        wget -nv -P /$WHEELHOUSE_DIR "http://mirrors.cambricon.com/pytorch/whl/test/torchaudio/${audio_whl}"
    else
        echo "Unknown install type: $version_type"
        exit 1
    fi
fi

########################################################
# Compile torch_mlu wheels
#######################################################
pip install /$WHEELHOUSE_DIR/torch-*.whl
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64/:${LD_LIBRARY_PATH}
pushd "$TORCH_MLU_HOME"
retry pip install -qr requirements.txt
python setup.py bdist_wheel -d /tmp/$WHEELHOUSE_DIR
mv /tmp/$WHEELHOUSE_DIR/torch_mlu*linux*.whl /$WHEELHOUSE_DIR/
popd
pip install /$WHEELHOUSE_DIR/torch_mlu*linux*.whl

if compgen -G "${TORCH_MLU_HOME}/pytorch_patches/*diff" > /dev/null; then
    if [[ "$PYTORCH_VERSION" == "main" ]] || [[ "$PYTORCH_VERSION" == "release"* ]]; then
        unset PYTORCH_VERSION    # This env may influence the deps of vision/audio
        yum update -y && yum install libjpeg-turbo-devel -y   # Only needed when compile torchvision
        ########################################################
        # Compile torchvision wheels
        #######################################################
        pushd "/vision"
        python setup.py bdist_wheel -d /tmp/$WHEELHOUSE_DIR
        mv /tmp/$WHEELHOUSE_DIR/torchvision*linux*.whl /$WHEELHOUSE_DIR/
        popd
        pip install /$WHEELHOUSE_DIR/torchvision*linux*.whl

        ########################################################
        # Compile torchaudio wheels
        #######################################################
        pushd "/audio"
        python setup.py bdist_wheel -d /tmp/$WHEELHOUSE_DIR
        mv /tmp/$WHEELHOUSE_DIR/torchaudio*linux*.whl /$WHEELHOUSE_DIR/
        popd
        pip install /$WHEELHOUSE_DIR/torchaudio*linux*.whl
        ########################################################
        # Smoke tests
        #######################################################
        python -c 'import torch; import torch_mlu; import torchvision; import torchaudio; print(torch.randn(3))'
    else
        ########################################################
        # Smoke tests
        #######################################################
        python -c 'import torch; import torch_mlu; print(torch.randn(3))'
    fi
else
    pip install /$WHEELHOUSE_DIR/torchvision-*.whl
    pip install /$WHEELHOUSE_DIR/torchaudio-*.whl
    ########################################################
    # Smoke tests
    #######################################################
    python -c 'import torch; import torch_mlu; import torchvision; import torchaudio; print(torch.randn(3))'
fi
