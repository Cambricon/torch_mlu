#!/bin/bash

find ${PWD}/. -name '.git' | xargs rm -rf; &>/dev/null
find ${PWD}/. -name '.jenkins*' | xargs rm -rf; &>/dev/null

rm -rf torch_mlu/docs &>/dev/null
rm -rf torch_mlu/pytorch_patches &>/dev/null
rm -rf torch_mlu/scripts/apply_patches_to_pytorch.sh &>/dev/null
rm -rf torch_mlu/scripts/build_torch_mlu.sh &>/dev/null
