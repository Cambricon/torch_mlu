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

// permute behaviour is same with cpu side.
/* example:
  contiguous tensor input with sizes (2, 3, 4, 2), strides (24 ,8 ,2 ,1);
  std::vector<int64> permute({0, 2, 3, 1});
  temp_output = at::permute(input, permute);
  output = cnnl_contiguous(temp_output, MemoryFormat);
  detail:
    temp_output is not contigous, and sizes (2, 4, 2, 3) and strides (24, 2, 1,
  8); if u need contiguous tensor with special MemoryFormat, need using like:
    output = at::permute(input, permute).contiguous(MemoryFormat);
    Python side:
      >>> a.size() original tensor
      torch.Size([2, 3, 4, 2])
      >>> a.stride()
      (24, 8, 2, 1)
      >>> b.size() permute tensor
      torch.Size([2, 4, 2, 3])
      >>> b.stride()
      (24, 2, 1, 8)
      >>> c.size() b.contiguous()
      torch.Size([2, 4, 2, 3])
      >>> c.stride()
      (24, 6, 3, 1)
      >>> d.size() b.contiguous(memory_format=torch.channels_last)
      torch.Size([2, 4, 2, 3])
      >>> d.stride()
      (24, 1, 12, 4)
*/

inline static void call_permute_without_any_check(
    at::Tensor& output,
    const at::Tensor& self,
    const std::vector<int>& cnnl_permute) {
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();

  // get cnnl descriptor
  cnnlDataType_t data_type = getCnnlType(input_impl);
  auto input_desc = getTensorDesc(input_impl, data_type, CNNL_LAYOUT_ARRAY);
  auto output_desc = getTensorDesc(output_impl, data_type, CNNL_LAYOUT_ARRAY);

  CnnlTransposeDescriptor trans_desc;
  trans_desc.set(cnnl_permute.size(), cnnl_permute.data());

  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // Get workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(
      handle, input_desc.get(), trans_desc.desc(), &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlTranspose_v2(
      handle,
      trans_desc.desc(),
      input_desc.get(),
      input_ptr,
      output_desc.get(),
      output_ptr,
      workspace_ptr.get(),
      workspace_size));
}

at::Tensor& cnnl_permute_out_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef dims) {
  const int p_dims = self.dim();
  TORCH_CHECK(p_dims == dims.size(), "number of dims don't match in permute.");
  TORCH_CHECK(
      self.is_contiguous(c10::MemoryFormat::Contiguous),
      "Self tensor Only support channels first contiguous.");

  std::vector<int> cnnl_permute(p_dims, 0);
  for (int i = 0; i < p_dims; ++i) {
    cnnl_permute[i] = static_cast<int>(at::maybe_wrap_dim(dims[i], p_dims));
  }

  std::vector<int> sort_permute(p_dims, 0);
  std::iota(sort_permute.begin(), sort_permute.end(), 0);
  if (cnnl_permute == sort_permute) {
    cnnl_copy_internal(output, self);
    return output;
  }

  call_permute_without_any_check(output, self, cnnl_permute);
  return output;
}

at::Tensor cnnl_permute_internal(const at::Tensor& self, at::IntArrayRef dims) {
  int p_dims = self.dim();
  TORCH_CHECK(p_dims == dims.size(), "number of dims don't match in permute.");
  TORCH_CHECK(
      self.is_contiguous(c10::MemoryFormat::Contiguous),
      "Self tensor Only support channels first contiguous.");

  std::vector<int> cnnl_permute(p_dims, 0);
  for (int i = 0; i < p_dims; ++i) {
    cnnl_permute[i] = static_cast<int>(at::maybe_wrap_dim(dims[i], p_dims));
  }

  // output
  auto input_size = self.sizes();
  std::vector<int64_t> output_size(p_dims);
  for (int i = 0; i < p_dims; ++i) {
    output_size[i] = static_cast<int64_t>(input_size[cnnl_permute[i]]);
  }
  // output is CF contiguous
  auto output =
      at::empty(output_size, self.options(), c10::MemoryFormat::Contiguous);
  call_permute_without_any_check(output, self, cnnl_permute);
  return output;
}

} // namespace ops
} // namespace torch_mlu
