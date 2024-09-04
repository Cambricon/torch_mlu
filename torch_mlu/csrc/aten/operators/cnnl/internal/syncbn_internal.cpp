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

#define GET_LAYOUT_OFFSET 2

static std::vector<cnnlTensorLayout_t> supported_input_layout = {
    CNNL_LAYOUT_NC,
    CNNL_LAYOUT_NLC,
    CNNL_LAYOUT_NHWC,
    CNNL_LAYOUT_NDHWC};

std::tuple<at::Tensor, at::Tensor> cnnl_batch_norm_stats_internal(
    const at::Tensor& input,
    float epsilon) {
  auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  auto input_options = input.options().dtype(acc_type);
  int64_t channel_num = input.size(1);
  auto mean = at::empty({channel_num}, input_options);
  auto invstd = at::empty({channel_num}, input_options);
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor mean_desc;
  CnnlTensorDescriptor invstd_desc;
  // It is the invoker's responsibility to ensure input.dim() will not be out of
  // range
  cnnlTensorLayout_t layout =
      supported_input_layout[input.dim() - GET_LAYOUT_OFFSET];
  input_desc.set(input, layout);
  mean_desc.set(mean);
  invstd_desc.set(invstd);
  auto input_impl = getMluTensorImpl(input);
  auto mean_impl = getMluTensorImpl(mean);
  auto invstd_impl = getMluTensorImpl(invstd);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);
  auto handle = getCurrentHandle();

  // get workspace for SBN
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetSyncBatchNormStatsWorkspaceSize(
      handle, input_desc.desc(), &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlSyncBatchNormStats_v2(
      /* handle         */ handle,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* workspace      */ ws_ptr.get(),
      /* workspace_size */ workspace_size,
      /* eps            */ epsilon,
      /* mean_desc      */ mean_desc.desc(),
      /* mean           */ mean_ptr,
      /* invstd_desc    */ invstd_desc.desc(),
      /* invstd         */ invstd_ptr));

  return std::make_tuple(mean, invstd);
}

std::tuple<at::Tensor, at::Tensor>
cnnl_batch_norm_gather_stats_with_counts_internal(
    const at::Tensor& mean_all,
    const at::Tensor& invstd_all,
    const at::Tensor& count_all,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    float momentum,
    float epsilon) {
  auto input_options = mean_all.options();
  if (mean_all.scalar_type() == at::ScalarType::Half) {
    input_options = input_options.dtype(at::ScalarType::Float);
  }
  int64_t channel_num = mean_all.size(1);
  auto mean = at::zeros({channel_num}, input_options);
  auto invstd = at::empty({channel_num}, input_options);
  if (mean_all.numel() == 0) {
    invstd.fill_(NAN);
    return std::make_tuple(mean, invstd);
  }
  CnnlTensorDescriptor mean_all_desc;
  CnnlTensorDescriptor invstd_all_desc;
  CnnlTensorDescriptor running_mean_desc;
  CnnlTensorDescriptor running_var_desc;
  CnnlTensorDescriptor count_all_desc;
  CnnlTensorDescriptor mean_desc;
  CnnlTensorDescriptor invstd_desc;
  mean_all_desc.set(mean_all, CNNL_LAYOUT_NC);
  invstd_all_desc.set(invstd_all, CNNL_LAYOUT_NC);
  running_mean_desc.set(running_mean);
  running_var_desc.set(running_var);
  count_all_desc.set(count_all);
  mean_desc.set(mean);
  invstd_desc.set(invstd);
  auto mean_all_impl = getMluTensorImpl(mean_all);
  auto invstd_all_impl = getMluTensorImpl(invstd_all);
  auto count_all_impl = getMluTensorImpl(count_all);
  auto mean_impl = getMluTensorImpl(mean);
  auto invstd_impl = getMluTensorImpl(invstd);
  auto mean_all_ptr = mlu_data_ptr(mean_all_impl);
  auto invstd_all_ptr = mlu_data_ptr(invstd_all_impl);
  auto count_all_ptr = mlu_data_ptr(count_all_impl);
  void* running_mean_ptr = nullptr;
  void* running_var_ptr = nullptr;
  if (running_mean.defined()) {
    auto running_mean_impl = getMluTensorImpl(running_mean);
    auto running_var_impl = getMluTensorImpl(running_var);
    running_mean_ptr = mlu_data_ptr(running_mean_impl);
    running_var_ptr = mlu_data_ptr(running_var_impl);
  }
  auto mean_ptr = mlu_data_ptr(mean_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);
  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlSyncBatchNormGatherStatsWithCounts(
      /* handle                */ handle,
      /* mean_all_desc         */ mean_all_desc.desc(),
      /* mean_all              */ mean_all_ptr,
      /* invstd_all_desc       */ invstd_all_desc.desc(),
      /* invstd_all            */ invstd_all_ptr,
      /* moving_mean_desc      */ running_mean_desc.desc(),
      /* moving_mean           */ running_mean_ptr,
      /* moving_var_desc       */ running_var_desc.desc(),
      /* moving_var            */ running_var_ptr,
      /* momentum              */ momentum,
      /* eps                   */ epsilon,
      /* count_all_desc        */ count_all_desc.desc(),
      /* count_all             */ count_all_ptr,
      /* mean_desc             */ mean_desc.desc(),
      /* mean                  */ mean_ptr,
      /* invstd_desc           */ invstd_desc.desc(),
      /* invstd                */ invstd_ptr));

  return std::make_tuple(mean, invstd);
}

at::Tensor& cnnl_batch_norm_elemt_out_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd) {
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor bias_desc;
  CnnlTensorDescriptor mean_desc;
  CnnlTensorDescriptor invstd_desc;
  // It is the invoker's responsibility to ensure input.dim() will not be out of
  // range
  cnnlTensorLayout_t layout =
      supported_input_layout[input.dim() - GET_LAYOUT_OFFSET];
  input_desc.set(input, layout);
  output_desc.set(output, layout);
  weight_desc.set(weight);
  bias_desc.set(bias);
  mean_desc.set(mean);
  invstd_desc.set(invstd);
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto mean_impl = getMluTensorImpl(mean);
  auto invstd_impl = getMluTensorImpl(invstd);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);
  void* weight_ptr = nullptr;
  void* bias_ptr = nullptr;
  if (weight.defined()) {
    auto weight_impl = getMluTensorImpl(weight);
    auto bias_impl = getMluTensorImpl(bias);
    weight_ptr = mlu_data_ptr(weight_impl);
    bias_ptr = mlu_data_ptr(bias_impl);
  }
  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlSyncBatchNormElemt(
      /* handle      */ handle,
      /* x_desc      */ input_desc.desc(),
      /* x           */ input_ptr,
      /* mean_desc   */ mean_desc.desc(),
      /* mean        */ mean_ptr,
      /* invstd_desc */ invstd_desc.desc(),
      /* invstd      */ invstd_ptr,
      /* weight_desc */ weight_desc.desc(),
      /* weight      */ weight_ptr,
      /* bias_desc   */ bias_desc.desc(),
      /* bias        */ bias_ptr,
      /* y_desc      */ output_desc.desc(),
      /* y           */ output_ptr));

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl_batch_norm_backward_reduce_internal(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const at::Tensor& weight,
    const bool input_g,
    const bool weight_g,
    const bool bias_g) {
  at::Tensor sum_dy;
  at::Tensor sum_dy_xmu;
  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (input_g) {
    sum_dy = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    sum_dy_xmu = at::empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  int64_t channel_num = input.size(1);
  if (weight_g) {
    grad_weight = at::empty({channel_num}, weight.options());
  }
  if (bias_g) {
    grad_bias = at::empty({channel_num}, weight.options());
  }

  CnnlTensorDescriptor dout_desc;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor mean_desc;
  CnnlTensorDescriptor invstd_desc;
  CnnlTensorDescriptor dweight_desc;
  CnnlTensorDescriptor dbias_desc;
  CnnlTensorDescriptor sum_dy_desc;
  CnnlTensorDescriptor sum_dy_xmu_desc;
  // It is the invoker's responsibility to ensure input.dim() will not be out of
  // range
  cnnlTensorLayout_t layout =
      supported_input_layout[input.dim() - GET_LAYOUT_OFFSET];
  dout_desc.set(grad_out, layout);
  input_desc.set(input, layout);
  mean_desc.set(mean);
  invstd_desc.set(invstd);
  dweight_desc.set(grad_weight);
  dbias_desc.set(grad_bias);
  sum_dy_desc.set(sum_dy);
  sum_dy_xmu_desc.set(sum_dy_xmu);
  auto dout_impl = getMluTensorImpl(grad_out);
  auto input_impl = getMluTensorImpl(input);
  auto mean_impl = getMluTensorImpl(mean);
  auto invstd_impl = getMluTensorImpl(invstd);
  auto dout_ptr = mlu_data_ptr(dout_impl);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);
  void* dweight_ptr = nullptr;
  void* dbias_ptr = nullptr;
  void* sum_dy_ptr = nullptr;
  void* sum_dy_xmu_ptr = nullptr;
  if (weight_g) {
    auto dweight_impl = getMluTensorImpl(grad_weight);
    dweight_ptr = mlu_data_ptr(dweight_impl);
  }
  if (bias_g) {
    auto dbias_impl = getMluTensorImpl(grad_bias);
    dbias_ptr = mlu_data_ptr(dbias_impl);
  }
  if (input_g) {
    auto sum_dy_impl = getMluTensorImpl(sum_dy);
    auto sum_dy_xmu_impl = getMluTensorImpl(sum_dy_xmu);
    sum_dy_ptr = mlu_data_ptr(sum_dy_impl);
    sum_dy_xmu_ptr = mlu_data_ptr(sum_dy_xmu_impl);
  }
  auto handle = getCurrentHandle();

  // get workspace for SYBNReduce
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetSyncBatchnormBackwardReduceWorkspaceSize(
      handle, input_desc.desc(), &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlSyncBatchnormBackwardReduce_v2(
      /* handle            */ handle,
      /* desc_dz           */ dout_desc.desc(),
      /* dz                */ dout_ptr,
      /* desc_x            */ input_desc.desc(),
      /* x                 */ input_ptr,
      /* desc_mean         */ mean_desc.desc(),
      /* mean              */ mean_ptr,
      /* desc_invstd       */ invstd_desc.desc(),
      /* invstd            */ invstd_ptr,
      /* workspace         */ ws_ptr.get(),
      /* workspace_size    */ workspace_size,
      /* desc_dweight      */ dweight_desc.desc(),
      /* dweight           */ dweight_ptr,
      /* desc_dbias        */ dbias_desc.desc(),
      /* dbias             */ dbias_ptr,
      /* desc_sum_dy       */ sum_dy_desc.desc(),
      /* sum_dy            */ sum_dy_ptr,
      /* desc_sum_dy_xmu   */ sum_dy_xmu_desc.desc(),
      /* sum_dy_xmu        */ sum_dy_xmu_ptr,
      /* needs_input_grad0 */ input_g,
      /* needs_input_grad1 */ weight_g,
      /* needs_input_grad2 */ bias_g));

  return std::make_tuple(sum_dy, sum_dy_xmu, grad_weight, grad_bias);
}

at::Tensor cnnl_batch_norm_backward_elemt_internal(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const at::Tensor& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count) {
  int64_t dim = input.dim();
  auto memory_format = (3 < dim && dim < 6)
      ? get_channels_last_memory_format(dim)
      : LEGACY_CONTIGUOUS_MEMORY_FORMAT;
  auto dinput = at::empty_like(input, memory_format);
  CnnlTensorDescriptor dout_desc;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor mean_desc;
  CnnlTensorDescriptor invstd_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor sum_dy_desc;
  CnnlTensorDescriptor sum_dy_xmu_desc;
  CnnlTensorDescriptor count_desc;
  CnnlTensorDescriptor dinput_desc;
  // It is the invoker's responsibility to ensure input.dim() will not be out of
  // range
  cnnlTensorLayout_t layout =
      supported_input_layout[input.dim() - GET_LAYOUT_OFFSET];
  dout_desc.set(grad_out, layout);
  input_desc.set(input, layout);
  mean_desc.set(mean);
  invstd_desc.set(invstd);
  weight_desc.set(weight);
  sum_dy_desc.set(sum_dy);
  sum_dy_xmu_desc.set(sum_dy_xmu);
  count_desc.set(count);
  dinput_desc.set(dinput, layout);
  auto dout_impl = getMluTensorImpl(grad_out);
  auto input_impl = getMluTensorImpl(input);
  auto mean_impl = getMluTensorImpl(mean);
  auto invstd_impl = getMluTensorImpl(invstd);
  auto sum_dy_impl = getMluTensorImpl(sum_dy);
  auto sum_dy_xmu_impl = getMluTensorImpl(sum_dy_xmu);
  auto count_impl = getMluTensorImpl(count);
  auto dinput_impl = getMluTensorImpl(dinput);
  auto dout_ptr = mlu_data_ptr(dout_impl);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);
  void* weight_ptr = nullptr;
  if (weight.defined()) {
    auto weight_impl = getMluTensorImpl(weight);
    weight_ptr = mlu_data_ptr(weight_impl);
  }
  auto sum_dy_ptr = mlu_data_ptr(sum_dy_impl);
  auto sum_dy_xmu_ptr = mlu_data_ptr(sum_dy_xmu_impl);
  auto count_ptr = mlu_data_ptr(count_impl);
  auto dinput_ptr = mlu_data_ptr(dinput_impl);
  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlSyncBatchNormBackwardElemtV2(
      /* handle           */ handle,
      /* diff_y_desc      */ dout_desc.desc(),
      /* diff_y           */ dout_ptr,
      /* x_desc           */ input_desc.desc(),
      /* x                */ input_ptr,
      /* mean_desc        */ mean_desc.desc(),
      /* mean             */ mean_ptr,
      /* invstd_desc      */ invstd_desc.desc(),
      /* invstd           */ invstd_ptr,
      /* weight_desc      */ weight_desc.desc(),
      /* weight           */ weight_ptr,
      /* sum_dy_desc      */ sum_dy_desc.desc(),
      /* sum_dy           */ sum_dy_ptr,
      /* sum_dy_xmu_desc  */ sum_dy_xmu_desc.desc(),
      /* sum_dy_xmu       */ sum_dy_xmu_ptr,
      /* count_desc       */ count_desc.desc(),
      /* count            */ count_ptr,
      /* diff_x_desc      */ dinput_desc.desc(),
      /* diff_x           */ dinput_ptr));

  return dinput;
}

} // namespace ops
} // namespace torch_mlu
