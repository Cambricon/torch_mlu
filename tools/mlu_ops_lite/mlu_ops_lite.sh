#!/bin/bash

# Get the path of current script
CUR_DIR=$(cd $(dirname $0);pwd)
# Get the root path of torch_mlu
TORCH_MLU_HOME=$CUR_DIR/../../
# Get the path of build.property
PROPERTY_PATH=$TORCH_MLU_HOME/scripts/release/

MLU_OPS_LITE_TAG=$(python -c "import json;f=open('$PROPERTY_PATH/build.property','r');bp=json.load(f);print(bp['src_requires']['mluops-lite'][0]);f.close()")
MLU_OPS_LITE_VERSION=$(python -c "import json;f=open('$PROPERTY_PATH/build.property','r');bp=json.load(f);print(bp['src_requires']['mluops-lite'][1]);f.close()")
if [[ ${MLU_OPS_LITE_TAG} == "release" ]]; then
    # MLU_OPS_LITE_VERSION=x.y.z
    wget -nv http://gitlab.software.cambricon.com/neuware/mlu-ops/-/archive/v${MLU_OPS_LITE_VERSION}/mlu-ops-v${MLU_OPS_LITE_VERSION}.zip
    unzip -q "mlu-ops-v${MLU_OPS_LITE_VERSION}.zip" -d "$TORCH_MLU_HOME/third_party/"
    mv "$TORCH_MLU_HOME/third_party/mlu-ops-v${MLU_OPS_LITE_VERSION}" "$TORCH_MLU_HOME/third_party/mlu-ops"
    rm "mlu-ops-v${MLU_OPS_LITE_VERSION}.zip"
else
    # MLU_OPS_LITE_VERSION=commit_id
    wget -nv http://gitlab.software.cambricon.com/neuware/mlu-ops/-/archive/${MLU_OPS_LITE_VERSION}.zip
    unzip -q "${MLU_OPS_LITE_VERSION}.zip" -d "$TORCH_MLU_HOME/third_party/"
    mv "$TORCH_MLU_HOME/third_party/mlu-ops-${MLU_OPS_LITE_VERSION}" "$TORCH_MLU_HOME/third_party/mlu-ops"
    rm "${MLU_OPS_LITE_VERSION}.zip"
fi

cd $TORCH_MLU_HOME/tools/mlu_ops_lite

# run copy_files.py
python copy_files.py

# check python script is success
if [ $? -ne 0 ]; then
    echo "Failed to run copy_files.py"
    exit 1
fi


# apply bangc_helper_dtype.diff
patch $TORCH_MLU_HOME/torch_mlu/csrc/aten/operators/bang/mlu_ops_lite/bangc_helper_dtype.h < bangc_helper_dtype.diff

# check patch is success
if [ $? -ne 0 ]; then
    echo "Failed to apply patch: bangc_helper_dtype.diff"
    exit 1
fi


patch $TORCH_MLU_HOME/torch_mlu/csrc/aten/operators/bang/mlu_ops_lite/bangc_kernels.h < bangc_kernels.diff

if [ $? -ne 0 ]; then
    echo "Failed to apply patch: bangc_kernels.diff"
    exit 1
fi

echo "Copy mlu_ops_lite files and apply patches successfully!"

# rm unzip dir
rm -rf "$TORCH_MLU_HOME/third_party/mlu-ops"

