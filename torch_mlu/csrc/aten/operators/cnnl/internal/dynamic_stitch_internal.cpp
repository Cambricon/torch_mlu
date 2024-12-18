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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_dynamic_stitch_internal(
    at::Tensor& out,
    at::TensorList indices_list,
    at::TensorList data_list,
    int indices_num) {
  c10::SmallVector<tensorDescPtr_t> indices_tensordesc_list;
  c10::SmallVector<cnnlTensorDescriptor_t> indices_desc_list;
  c10::SmallVector<int*> indices_ptr_list;
  c10::SmallVector<tensorDescPtr_t> data_tensordesc_list;
  c10::SmallVector<cnnlTensorDescriptor_t> data_desc_list;
  c10::SmallVector<void*> data_ptr_list;
  c10::SmallVector<int> indices_dims;
  for (int input_num = 0; input_num < indices_list.size(); input_num++) {
    auto indices = indices_list[input_num];
    auto data = data_list[input_num];
    auto data_impl = getMluTensorImpl(data);
    auto data_ptr = data_impl->mlu_data_ptr();
    auto indices_impl = getMluTensorImpl(indices);
    auto indices_ptr = indices_impl->mlu_data_ptr();
    indices_tensordesc_list.emplace_back(
        getTensorDesc(indices_impl, CNNL_LAYOUT_ARRAY));
    indices_desc_list.emplace_back(indices_tensordesc_list.back().get());
    data_tensordesc_list.emplace_back(
        getTensorDesc(data_impl, CNNL_LAYOUT_ARRAY));
    data_desc_list.emplace_back(data_tensordesc_list.back().get());
    indices_ptr_list.emplace_back(reinterpret_cast<int*>(indices_ptr));
    data_ptr_list.emplace_back(data_ptr);
    indices_dims.emplace_back(indices.numel());
  }
  auto out_impl = getMluTensorImpl(out);
  auto out_desc = getTensorDesc(out_impl);
  auto out_ptr = out_impl->mlu_data_ptr();

  // get current handle
  auto handle = getCurrentHandle();

  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetDynamicStitchWorkspaceSize_v2(
      handle,
      indices_desc_list.data(),
      data_desc_list.data(),
      indices_list.size(),
      indices_dims.data(),
      &space_size,
      out_desc.get()));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  // calculate
  TORCH_CNNL_CHECK(cnnlDynamicStitch(
      handle,
      indices_desc_list.data(),
      const_cast<const int**>(indices_ptr_list.data()),
      data_desc_list.data(),
      const_cast<const void**>(data_ptr_list.data()),
      indices_list.size(),
      indices_dims.data(),
      workspace_ptr.get(),
      space_size,
      out_desc.get(),
      out_ptr));
  return;
}

} // namespace ops
} // namespace torch_mlu
