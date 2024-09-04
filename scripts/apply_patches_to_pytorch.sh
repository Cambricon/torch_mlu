#!/usr/bin/env bash
#set -e

function usage () {
    echo -e "\033[32m Usage : \033[0m"
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
    echo "|  ./apply_patches_to_pytorch.sh [OPTIONAL]"
    echo "|      Supported options:"
    echo "|             1. no file specified : apply all patches in ../pytorch_patches."
    echo "|             2. specify certain patches in ../pytorch_patches: only apply the patches specified."
    echo "|             3. '-h': show this help."
    echo "|                                                   "
    echo "|  eg.1. ./apply_patches_to_pytorch.sh"
    echo "|      this will apply all the patches in ../pytorch_patches."
    echo "|  eg.2. ./apply_patches_to_pytorch.sh patch1 pacth2"
    echo "|      this will only apply ../pytorch_patches/patch1 and ../pytorch_patches/pacth2."
    echo -e "\033[32m ------------------------------------------------------------------- \033[0m"
}

while getopts 'h' opt; do
    case "$opt" in
        h)  usage ; exit 1 ;;
        ?)  echo "unrecognized optional arg : "; $opt; usage; exit 1;;
    esac
done

# Get the path of current script
CUR_DIR=$(cd $(dirname $0);pwd)
# The default folder structure: torch_mlu/scripts/apply_patches_to_pytorch.sh
PYTORCH_REPO_ROOT=$CUR_DIR/../../
# Get the path of pytorch patches
PATCHES_DIR=$CUR_DIR/../pytorch_patches/

all_patch_files=""
function get_all_patches() {
    for file in $(ls $1)
    do
        if [ -d $1/$file ];then
            get_all_patches $1/$file
        else
            if [ "${file##*.}"x = "diff"x ];then
                all_patch_files+="$file "
            fi
        fi
    done
}

get_all_patches $PATCHES_DIR

# if specify certain patch file, check the patch file name
if [ $# -ge 1 ];then
    patch_files=$@
    for file in $patch_files
    do
      if [[ $all_patch_files =~ $file ]];then
          continue
      else
          echo -e "\033[31m ERROR: \033[0m Patch file '$file' not in ../pytorch_patches."
          usage
          exit 1
      fi
    done
else
    patch_files=$all_patch_files
fi

# If PYTORCH_HOME Env is set, use it.
if [ -n "${PYTORCH_HOME}" ];then
    PYTORCH_REPO_ROOT=${PYTORCH_HOME}
fi

echo "PYTORCH_HOME: $PYTORCH_REPO_ROOT"
echo "PYTORCH_PATCHES_DIR: $PATCHES_DIR"

commit_file=`cat $PATCHES_DIR/commit_id`
for commit_id in $commit_file
do
    if [[ $commit_id =~ "pytorch" ]];then
        ORIG_PYTORCH_COMMIT_ID=${commit_id#*:}
    fi
    if [[ $commit_id =~ "kineto" ]];then
        KINETO_COMMIT_ID=${commit_id#*:}
    fi
done
# Clean Pytorch environment when .git exists in Pytorch before patching
if [ -d "$PYTORCH_REPO_ROOT/.git" ];then
    echo "Checking PyTorch HEAD."
    pushd $PYTORCH_REPO_ROOT
    CUR_HEAD=`git rev-parse HEAD@{0}`
    popd
    if [[ -n "$ORIG_PYTORCH_COMMIT_ID" && $CUR_HEAD != $ORIG_PYTORCH_COMMIT_ID ]];then
        echo -e "\033[31m WARNING: Pytorch HEAD ('${CUR_HEAD}') has changed which means your pytorch version ('${ORIG_PYTORCH_COMMIT_ID}') may not be consistent with Cambricon Neuware. This may cause ambiguity! Please make sure you know what you are doing. \033[0m"
        read -p "Still want to continue? [Y/n]:" input
        case $input in
            [Yy]) ;;
            [Nn]) exit 1 ;;
            ?)  echo "unrecognized arg : "; $input; exit 1;;
            *)  exit 1;;
        esac
    fi
    echo "Cleaning the Pytorch Environment before patching."
    pushd $PYTORCH_REPO_ROOT
    git checkout .
    rm -rf $PYTORCH_REPO_ROOT/torch/mlu
    popd
else
    echo -e "\033[33mThere is no .git directory in PyTorch source. Patch may cause error...\033[0m"
fi

# The setting args of patch commond
patch_args="-p 1 -l -N -s --no-backup-if-mismatch"

# Apply patches into Pytorch
for file in $patch_files
do
    if [[ $file =~ 'kineto' ]];then
        # Apply kineto patch into pytorch/third_party/kineto
        KINETO_REPO_ROOT=$PYTORCH_REPO_ROOT/third_party/kineto
        KINETO_PATCH_ROOT=$PATCHES_DIR/third_party/kineto
        if [ -z "$(ls -A $KINETO_REPO_ROOT)" ];then
            echo -e "\033[31m ERROR: Need to update pytorch submodule 'kineto' first. \033[0m"
            exit -1
        fi
        if [ -f "$KINETO_REPO_ROOT/.git" ];then
            pushd $KINETO_REPO_ROOT
            echo "Checking Kineto HEAD."
            KINETO_CUR_HEAD=`git rev-parse HEAD@{0}`
            if [[ -n "$KINETO_COMMIT_ID" && $KINETO_CUR_HEAD != $KINETO_COMMIT_ID ]];then
        	echo -e "\033[31m WARNING: Kineto HEAD has changed. Applying patch to kineto may be failed. \033[0m"
            fi
            git checkout .
            popd
        fi
        echo "Apply patch: $file"
        patch -d $KINETO_REPO_ROOT -i $KINETO_PATCH_ROOT/$file $patch_args
    else
      echo "Apply patch: $file"
      patch -d $PYTORCH_REPO_ROOT -i $PATCHES_DIR/$file $patch_args
    fi
done
