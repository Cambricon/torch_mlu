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

static const std::vector<cnnlTensorLayout_t> supported_input_layout = {
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

  // It is the invoker's responsibility to ensure input.dim() will not be out of
  // range
  cnnlTensorLayout_t layout =
      supported_input_layout[input.dim() - GET_LAYOUT_OFFSET];

  auto input_impl = getMluTensorImpl(input);
  auto mean_impl = getMluTensorImpl(mean);
  auto invstd_impl = getMluTensorImpl(invstd);

  auto input_desc = getTensorDesc(input_impl, layout);
  auto mean_desc = getTensorDesc(mean_impl);
  auto invstd_desc = getTensorDesc(invstd_impl);

  auto input_ptr = mlu_data_ptr(input_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);

  auto handle = getCurrentHandle();

  // get workspace for SBN
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetSyncBatchNormStatsWorkspaceSize(
      handle, input_desc.get(), &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlSyncBatchNormStats_v2(
      /* handle         */ handle,
      /* x_desc         */ input_desc.get(),
      /* x              */ input_ptr,
      /* workspace      */ ws_ptr.get(),
      /* workspace_size */ workspace_size,
      /* eps            */ epsilon,
      /* mean_desc      */ mean_desc.get(),
      /* mean           */ mean_ptr,
      /* invstd_desc    */ invstd_desc.get(),
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

  auto mean_all_impl = getMluTensorImpl(mean_all);
  auto mean_all_desc = getTensorDesc(mean_all_impl, CNNL_LAYOUT_NC);
  auto mean_all_ptr = mlu_data_ptr(mean_all_impl);

  auto invstd_all_impl = getMluTensorImpl(invstd_all);
  auto invstd_all_desc = getTensorDesc(invstd_all_impl, CNNL_LAYOUT_NC);
  auto invstd_all_ptr = mlu_data_ptr(invstd_all_impl);

  auto count_all_impl = getMluTensorImpl(count_all);
  auto count_all_desc = getTensorDesc(count_all_impl);
  auto count_all_ptr = mlu_data_ptr(count_all_impl);

  auto mean_impl = getMluTensorImpl(mean);
  auto mean_desc = getTensorDesc(mean_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);

  auto invstd_impl = getMluTensorImpl(invstd);
  auto invstd_desc = getTensorDesc(invstd_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);

  void* running_mean_ptr = nullptr;
  void* running_var_ptr = nullptr;

  tensorDescPtr_t running_mean_desc = nullptr;
  tensorDescPtr_t running_var_desc = nullptr;
  if (running_mean.defined()) {
    auto running_mean_impl = getMluTensorImpl(running_mean);
    auto running_var_impl = getMluTensorImpl(running_var);

    running_mean_desc = getTensorDesc(running_mean_impl);
    running_var_desc = getTensorDesc(running_var_impl);

    running_mean_ptr = mlu_data_ptr(running_mean_impl);
    running_var_ptr = mlu_data_ptr(running_var_impl);
  }

  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlSyncBatchNormGatherStatsWithCounts(
      /* handle                */ handle,
      /* mean_all_desc         */ mean_all_desc.get(),
      /* mean_all              */ mean_all_ptr,
      /* invstd_all_desc       */ invstd_all_desc.get(),
      /* invstd_all            */ invstd_all_ptr,
      /* moving_mean_desc      */ running_mean_desc.get(),
      /* moving_mean           */ running_mean_ptr,
      /* moving_var_desc       */ running_var_desc.get(),
      /* moving_var            */ running_var_ptr,
      /* momentum              */ momentum,
      /* eps                   */ epsilon,
      /* count_all_desc        */ count_all_desc.get(),
      /* count_all             */ count_all_ptr,
      /* mean_desc             */ mean_desc.get(),
      /* mean                  */ mean_ptr,
      /* invstd_desc           */ invstd_desc.get(),
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
  // It is the invoker's responsibility to ensure input.dim() will not be out of
  // range
  cnnlTensorLayout_t layout =
      supported_input_layout[input.dim() - GET_LAYOUT_OFFSET];
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto mean_impl = getMluTensorImpl(mean);
  auto invstd_impl = getMluTensorImpl(invstd);

  auto input_desc = getTensorDesc(input_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);
  auto mean_desc = getTensorDesc(mean_impl);
  auto invstd_desc = getTensorDesc(invstd_impl);

  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);

  void* weight_ptr = nullptr;
  void* bias_ptr = nullptr;
  tensorDescPtr_t weight_desc = nullptr;
  tensorDescPtr_t bias_desc = nullptr;
  if (weight.defined()) {
    auto weight_impl = getMluTensorImpl(weight);
    weight_ptr = mlu_data_ptr(weight_impl);
    weight_desc = getTensorDesc(weight_impl);

    auto bias_impl = getMluTensorImpl(bias);
    bias_ptr = mlu_data_ptr(bias_impl);
    bias_desc = getTensorDesc(bias_impl);
  }
  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlSyncBatchNormElemt(
      /* handle      */ handle,
      /* x_desc      */ input_desc.get(),
      /* x           */ input_ptr,
      /* mean_desc   */ mean_desc.get(),
      /* mean        */ mean_ptr,
      /* invstd_desc */ invstd_desc.get(),
      /* invstd      */ invstd_ptr,
      /* weight_desc */ weight_desc.get(),
      /* weight      */ weight_ptr,
      /* bias_desc   */ bias_desc.get(),
      /* bias        */ bias_ptr,
      /* y_desc      */ output_desc.get(),
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

  // It is the invoker's responsibility to ensure input.dim() will not be out of
  // range
  cnnlTensorLayout_t layout =
      supported_input_layout[input.dim() - GET_LAYOUT_OFFSET];

  auto dout_impl = getMluTensorImpl(grad_out);
  auto dout_desc = getTensorDesc(dout_impl, layout);
  auto dout_ptr = mlu_data_ptr(dout_impl);

  auto input_impl = getMluTensorImpl(input);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto mean_impl = getMluTensorImpl(mean);
  auto mean_desc = getTensorDesc(mean_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);

  auto invstd_impl = getMluTensorImpl(invstd);
  auto invstd_desc = getTensorDesc(invstd_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);

  void* dweight_ptr = nullptr;
  void* dbias_ptr = nullptr;
  void* sum_dy_ptr = nullptr;
  void* sum_dy_xmu_ptr = nullptr;
  tensorDescPtr_t dweight_desc = nullptr;
  tensorDescPtr_t dbias_desc = nullptr;
  tensorDescPtr_t sum_dy_desc = nullptr;
  tensorDescPtr_t sum_dy_xmu_desc = nullptr;

  if (weight_g) {
    auto dweight_impl = getMluTensorImpl(grad_weight);
    dweight_ptr = mlu_data_ptr(dweight_impl);
    dweight_desc = getTensorDesc(dweight_impl);
  }
  if (bias_g) {
    auto dbias_impl = getMluTensorImpl(grad_bias);
    dbias_ptr = mlu_data_ptr(dbias_impl);
    dbias_desc = getTensorDesc(dbias_impl);
  }
  if (input_g) {
    auto sum_dy_impl = getMluTensorImpl(sum_dy);
    sum_dy_ptr = mlu_data_ptr(sum_dy_impl);
    sum_dy_desc = getTensorDesc(sum_dy_impl);

    auto sum_dy_xmu_impl = getMluTensorImpl(sum_dy_xmu);
    sum_dy_xmu_ptr = mlu_data_ptr(sum_dy_xmu_impl);
    sum_dy_xmu_desc = getTensorDesc(sum_dy_xmu_impl);
  }
  auto handle = getCurrentHandle();

  // get workspace for SYBNReduce
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetSyncBatchnormBackwardReduceWorkspaceSize(
      handle, input_desc.get(), &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlSyncBatchnormBackwardReduce_v2(
      /* handle            */ handle,
      /* desc_dz           */ dout_desc.get(),
      /* dz                */ dout_ptr,
      /* desc_x            */ input_desc.get(),
      /* x                 */ input_ptr,
      /* desc_mean         */ mean_desc.get(),
      /* mean              */ mean_ptr,
      /* desc_invstd       */ invstd_desc.get(),
      /* invstd            */ invstd_ptr,
      /* workspace         */ ws_ptr.get(),
      /* workspace_size    */ workspace_size,
      /* desc_dweight      */ dweight_desc.get(),
      /* dweight           */ dweight_ptr,
      /* desc_dbias        */ dbias_desc.get(),
      /* dbias             */ dbias_ptr,
      /* desc_sum_dy       */ sum_dy_desc.get(),
      /* sum_dy            */ sum_dy_ptr,
      /* desc_sum_dy_xmu   */ sum_dy_xmu_desc.get(),
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

  // It is the invoker's responsibility to ensure input.dim() will not be out of
  // range
  cnnlTensorLayout_t layout =
      supported_input_layout[input.dim() - GET_LAYOUT_OFFSET];

  auto dout_impl = getMluTensorImpl(grad_out);
  auto dout_desc = getTensorDesc(dout_impl, layout);
  auto dout_ptr = mlu_data_ptr(dout_impl);

  auto input_impl = getMluTensorImpl(input);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto mean_impl = getMluTensorImpl(mean);
  auto mean_desc = getTensorDesc(mean_impl);
  auto mean_ptr = mlu_data_ptr(mean_impl);

  auto invstd_impl = getMluTensorImpl(invstd);
  auto invstd_desc = getTensorDesc(invstd_impl);
  auto invstd_ptr = mlu_data_ptr(invstd_impl);

  auto sum_dy_impl = getMluTensorImpl(sum_dy);
  auto sum_dy_desc = getTensorDesc(sum_dy_impl);
  auto sum_dy_ptr = mlu_data_ptr(sum_dy_impl);

  auto sum_dy_xmu_impl = getMluTensorImpl(sum_dy_xmu);
  auto sum_dy_xmu_desc = getTensorDesc(sum_dy_xmu_impl);
  auto sum_dy_xmu_ptr = mlu_data_ptr(sum_dy_xmu_impl);

  auto count_impl = getMluTensorImpl(count);
  auto count_desc = getTensorDesc(count_impl);
  auto count_ptr = mlu_data_ptr(count_impl);

  auto dinput_impl = getMluTensorImpl(dinput);
  auto dinput_desc = getTensorDesc(dinput_impl, layout);
  auto dinput_ptr = mlu_data_ptr(dinput_impl);

  void* weight_ptr = nullptr;
  tensorDescPtr_t weight_desc = nullptr;
  if (weight.defined()) {
    auto weight_impl = getMluTensorImpl(weight);
    weight_desc = getTensorDesc(weight_impl);
    weight_ptr = mlu_data_ptr(weight_impl);
  }
  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlSyncBatchNormBackwardElemtV2(
      /* handle           */ handle,
      /* diff_y_desc      */ dout_desc.get(),
      /* diff_y           */ dout_ptr,
      /* x_desc           */ input_desc.get(),
      /* x                */ input_ptr,
      /* mean_desc        */ mean_desc.get(),
      /* mean             */ mean_ptr,
      /* invstd_desc      */ invstd_desc.get(),
      /* invstd           */ invstd_ptr,
      /* weight_desc      */ weight_desc.get(),
      /* weight           */ weight_ptr,
      /* sum_dy_desc      */ sum_dy_desc.get(),
      /* sum_dy           */ sum_dy_ptr,
      /* sum_dy_xmu_desc  */ sum_dy_xmu_desc.get(),
      /* sum_dy_xmu       */ sum_dy_xmu_ptr,
      /* count_desc       */ count_desc.get(),
      /* count            */ count_ptr,
      /* diff_x_desc      */ dinput_desc.get(),
      /* diff_x           */ dinput_ptr));

  return dinput;
}

} // namespace ops
} // namespace torch_mlu
