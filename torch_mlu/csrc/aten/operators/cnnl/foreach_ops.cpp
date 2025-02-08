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
#include <ATen/native/ForeachUtils.h>

#include "aten/utils/foreach_check_utils.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/foreach_common_utils.h"

namespace torch_mlu {
namespace ops {

typedef std::pair<std::vector<at::Tensor>, std::vector<at::Tensor>> copy_pair;

inline bool can_use_fast_route(
    at::ArrayRef<at::TensorList> tensorLists,
    at::ArrayRef<at::Scalar> scalarList = {},
    bool does_op_promote_integer_inputs_to_float = false) {
  return at::native::_check_tensors_share_device_and_dtype(tensorLists) &&
      at::native::_check_tensors_do_type_promotion_with_scalars(
             tensorLists[0],
             scalarList,
             does_op_promote_integer_inputs_to_float);
}

std::map<size_t, copy_pair> process_input_params(
    const at::TensorList& self,
    const at::TensorList& src) {
  auto is_diff_stride = [](const IntArrayRef& size,
                           const IntArrayRef& left_stride,
                           const IntArrayRef& right_stride) -> bool {
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

  std::map<size_t, copy_pair> copy_map;
  for (const auto i : c10::irange(self.size())) {
    if (self[i].dtype() != src[i].dtype()) {
      CNLOG(INFO) << "src and dst has different dtype"
                  << " dst dtype " << self[i].dtype() << " and src dtype "
                  << src[i].dtype();
      copy_map[0].first.push_back(self[i]);
      copy_map[0].second.push_back(src[i]);
    } else if (self[i].sizes() != src[i].sizes()) {
      CNLOG(INFO) << "src and dst has different sizes"
                  << " dst sizes " << self[i].sizes() << " and src sizes "
                  << src[i].sizes();
      copy_map[0].first.push_back(self[i]);
      copy_map[0].second.push_back(src[i]);
    } else if (is_diff_stride(
                   self[i].sizes(), self[i].strides(), src[i].strides())) {
      CNLOG(INFO) << "src and dst has different strides"
                  << " dst strides " << self[i].strides() << " and src strides "
                  << src[i].strides();
      copy_map[0].first.push_back(self[i]);
      copy_map[0].second.push_back(src[i]);
    } else {
      size_t data_type_bits = self[i].element_size();
      copy_map[data_type_bits].first.push_back(self[i]);
      copy_map[data_type_bits].second.push_back(src[i]);
    }
  }
  return copy_map;
}
void cnnl__foreach_copy_(
    at::TensorList self,
    at::TensorList src,
    const bool non_blocking) {
  at::native::check_foreach_api_restrictions(self, src);
  if (!torch_mlu::can_use_fast_route(
          self, src, /* does_op_promote_integer_inputs_to_float */ false)) {
    at::native::foreach_tensor_copy_list_kernel_slow_(self, src, non_blocking);
    for (const auto i : c10::irange(self.size())) {
      self[i].copy_(src[i], non_blocking);
    }
  } else {
    auto copy_map = process_input_params(self, src);
    for (const auto& pair : copy_map) {
      if (pair.first == 0) {
        at::native::foreach_tensor_copy_list_kernel_slow_(
            pair.second.first, pair.second.second, non_blocking);
      } else {
        auto handle = getCurrentHandle();
        ForeachOPTensorScalarHandle<1, 1, false> tensor_desc_ptr(
            {pair.second.second, pair.second.first}, {});
        const int64_t tensor_num = tensor_desc_ptr.get_tensor_num();
        if (tensor_num == 0)
          return;
        auto [input_desc_array, input_ptr_array] =
            tensor_desc_ptr.template get_input_tensor_desc_and_ptr<0>();
        auto [output_desc_array, output_ptr_array] =
            tensor_desc_ptr.template get_output_tensor_desc_and_ptr<0>();
        TORCH_CNNL_CHECK(cnnlForeachCopy(
            handle,
            tensor_num,
            input_desc_array,
            input_ptr_array,
            output_desc_array,
            output_ptr_array));
      }
    }
  }
}
} // namespace ops
} // namespace torch_mlu
