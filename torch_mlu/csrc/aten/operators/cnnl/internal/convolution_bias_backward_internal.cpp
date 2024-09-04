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
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

// Only used for conv bias backward.
at::Tensor& cnnl_bias_backward_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim) {
  const int input_dim = input.dim();
  TORCH_CHECK(
      input_dim == 4 || input_dim == 5,
      "cnnl bias backward only support 4d or 5d input tensor.");
  TORCH_CHECK(dim == 1, "cnnl bias backward dim only support value 1.");
  auto memory_format = get_channels_last_memory_format(input_dim);
  TORCH_CHECK(
      input.is_contiguous(memory_format), "input must be CL contiguous.");
  auto layout = input_dim > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  int64_t dim_internal = input_dim > 4 ? 4 : 3;
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();

  auto desc_input = getTensorDesc(input_impl, layout);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get workspace for BiasAddBackward
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetBiasAddBackwardWorkspaceSize(
      handle,
      desc_input.get(),
      desc_output.get(),
      dim_internal,
      &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // compute
  TORCH_CNNL_CHECK(cnnlBiasAddBackward_v2(
      handle,
      desc_input.get(),
      input_ptr,
      dim_internal,
      desc_output.get(),
      output_ptr,
      ws_ptr.get(),
      workspace_size));
  return output;
}
} // namespace ops
} // namespace torch_mlu
