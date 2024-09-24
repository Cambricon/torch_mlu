/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include "ATen/native/ForeachUtils.h"

// This file is copied from pytorch ForeachUtils.h. URL:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ForeachUtils.h
// Only add patch for fix a bug, which is to fix stride compare failed when
// size value equal to one in function _check_tensors_share_device_and_dtype.
// Bug fix mr: https://github.com/pytorch/pytorch/pull/134546
// Affect API list: check_fast_path_restrictions,
// _check_tensors_share_device_and_dtype,
//                  can_use_fast_route.

namespace torch_mlu {

// Helper function called in check_fast_path_restrictions to check if
// corresponding tensors in tensor lists have the same sizes and strides.
inline bool _check_tensors_share_sizes_and_strides(
    at::ArrayRef<at::TensorList> tensorLists) {
  // merged in pytorch main branch.
  // https://github.com/pytorch/pytorch/pull/134546
  auto is_diff_stride = [](const at::IntArrayRef& size,
                           const at::IntArrayRef& left_stride,
                           const at::IntArrayRef& right_stride) -> bool {
    const size_t size_size = size.size();
    for (const auto dim : c10::irange(size_size)) {
      if (size[dim] == 1)
        continue;
      if (left_stride[dim] != right_stride[dim]) {
        return true;
      }
    }
    return false;
  };
  for (const auto i : c10::irange(1, tensorLists.size())) {
    for (const auto j : c10::irange(tensorLists[0].size())) {
      if (tensorLists[0][j].sizes() != tensorLists[i][j].sizes() ||
          is_diff_stride(
              tensorLists[0][j].sizes(),
              tensorLists[0][j].strides(),
              tensorLists[i][j].strides())) {
        return false;
      }
    }
  }

  return true;
}

// To go via 'fast' path, several conditions must be satisfied
// - All tensors in all lists must have the same dtype.
// - All tensors must be on the same device
// - All tensors must have strided layout
// - All tensors must be non-overlapping and dense
// - Resulting tensor must have the same dtype as the input one

// [note: what's ``does_op_promote_integer_inputs_to_float=true``?]
//     ``does_op_promote_integer_inputs_to_float=true`` means that the result of
//     the op will be float even if inputs are integer or boolean, which
//     currently fast path does not support. In short, this flag, when
//     turned on, gatekeeps the op from going down the fastpath.

// Please, make sure to call check_foreach_api_restrictions before calling this
// method. There is a set of preconditions that have to be satisfied.
inline bool check_fast_path_restrictions(
    at::ArrayRef<at::TensorList> tensorLists,
    at::ArrayRef<at::Scalar> scalarList = {},
    bool does_op_promote_integer_inputs_to_float = false) {
  return at::native::_check_tensors_share_device_and_dtype(tensorLists) &&
      torch_mlu::_check_tensors_share_sizes_and_strides(tensorLists) &&
      at::native::_check_tensors_do_type_promotion_with_scalars(
             tensorLists[0],
             scalarList,
             does_op_promote_integer_inputs_to_float);
}

// see: [note: what's ``does_op_promote_integer_inputs_to_float=true``?]
inline bool can_use_fast_route(
    at::ArrayRef<at::TensorList> tensorLists,
    at::ArrayRef<at::Scalar> scalarList = {},
    bool does_op_promote_integer_inputs_to_float = false) {
  return torch_mlu::check_fast_path_restrictions(
      tensorLists, scalarList, does_op_promote_integer_inputs_to_float);
}

// see: [note: what's ``does_op_promote_integer_inputs_to_float=true``?]
inline bool can_use_fast_route(
    at::TensorList tensors1,
    at::TensorList tensors2,
    bool does_op_promote_integer_inputs_to_float = false) {
  return torch_mlu::can_use_fast_route(
      {tensors1, tensors2}, {}, does_op_promote_integer_inputs_to_float);
}

} // namespace torch_mlu