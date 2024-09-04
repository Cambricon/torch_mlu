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

at::Tensor& col2im_out_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& stride) {
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_output;
  CnnlTensorDescriptor weight_desc;
  CnnlConvolutionDescriptor conv_desc;

  // get cnnl descriptor
  desc_input.set(input, CNNL_LAYOUT_ARRAY);
  // Just for pass filter kernel info.
  std::vector<int64_t> weight_sizes{1, 1, kernel_size[0], kernel_size[1]};
  std::vector<int64_t> weight_strides{1, 1, 1, 1};
  auto output_dtype = getCnnlDataType(output.dtype());
  weight_desc.set(
      input, weight_sizes, weight_strides, CNNL_LAYOUT_NCHW, output_dtype);
  desc_output.set(output, CNNL_LAYOUT_NCHW);
  const int64_t groups = 1;
  const int64_t fake_input_dim = 4;
  conv_desc.set(
      fake_input_dim,
      stride.data(),
      padding.data(),
      dilation.data(),
      groups,
      output_dtype,
      false);

  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetCol2ImWorkspaceSize(
      handle,
      desc_input.desc(),
      weight_desc.desc(),
      desc_output.desc(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlCol2Im(
      handle,
      desc_input.desc(),
      input_ptr,
      weight_desc.desc(),
      conv_desc.desc(),
      workspace_ptr.get(),
      workspace_size,
      desc_output.desc(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
