/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
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

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_group_norm_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& mean,
    at::Tensor& rstd,
    double eps,
    int64_t num_groups) {
  if (input.numel() == 0) {
    return std::make_tuple(output, mean, rstd);
  }
  auto input_impl = getMluTensorImpl(input);
  auto weight_impl = getMluTensorImpl(weight);
  auto bias_impl = getMluTensorImpl(bias);
  auto mean_impl = getMluTensorImpl(mean);
  auto rstd_impl = getMluTensorImpl(rstd);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor weight_bias_desc;
  CnnlTensorDescriptor mean_rstd_desc;

  auto layout = suggest_cnnl_layout(input);
  input_desc.set(input, layout);
  output_desc.set(output, layout);
  weight_bias_desc.set(weight);
  mean_rstd_desc.set(mean);

  auto input_ptr = input_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto bias_ptr = bias_impl->mlu_data_ptr();
  auto mean_ptr = mean_impl->mlu_data_ptr();
  auto rstd_ptr = rstd_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // get workspace for GroupNormForward
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetGroupNormForwardWorkspaceSize(
      handle, num_groups, input_desc.desc(), &workspace_size));

  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  TORCH_CNNL_CHECK(cnnlGroupNormForward_v3(
      handle,
      eps,
      num_groups,
      input_desc.desc(),
      input_ptr,
      weight_bias_desc.desc(),
      weight_ptr,
      bias_ptr,
      ws_ptr.get(),
      workspace_size,
      output_desc.desc(),
      output_ptr,
      mean_rstd_desc.desc(),
      mean_ptr,
      rstd_ptr));
  return std::make_tuple(output, mean, rstd);
}

} // namespace ops
} // namespace torch_mlu
