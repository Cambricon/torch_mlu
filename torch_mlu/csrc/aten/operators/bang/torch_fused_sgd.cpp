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

#include "aten/utils/foreach_check_utils.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

namespace torch_mlu {
namespace ops {

template <int depth>
void _fused_sgd_mlu_internal_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const double lr,
    const float* lr_ptr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  const int tensor_num = params.size();

  auto stream = getCurMLUStream();
  const int64_t device_index = params[0].get_device();
  // compute kernel dim
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  k_dim.x = torch_mlu::getDeviceProperties(device_index)->core_num_per_cluster;
  k_dim.y = torch_mlu::getDeviceProperties(device_index)->cluster_count;
  k_dim.z = 1;
  const int nram_size = torch_mlu::getDeviceProperties(device_index)->nram_size;
  // get grad and param tensor cnrt type
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      params[0].scalar_type(),
      "fused_sgd_mlu",
      [&]() {
        // avoid call getMluTensorImpl in bangc utils code
        std::vector<std::array<void*, depth>> tensors_ptr_list;
        std::vector<int64_t> tensor_sizes_list;
        tensors_ptr_list.reserve(tensor_num);
        tensor_sizes_list.reserve(tensor_num);
        for (int i = 0; i < tensor_num; ++i) {
          const at::Tensor& param = params[i];
          const int num_elements = param.numel();
          if (num_elements == 0) {
            continue;
          }
          tensor_sizes_list.push_back(num_elements);
          std::array<void*, depth> tensors_ptr_array;
          if constexpr (depth == 3) {
            tensors_ptr_array = {
                mlu_data_ptr(getMluTensorImpl(param)),
                mlu_data_ptr(getMluTensorImpl(grads[i])),
                mlu_data_ptr(getMluTensorImpl(momentum_buffer_list[i]))};
          } else {
            tensors_ptr_array = {
                mlu_data_ptr(getMluTensorImpl(param)),
                mlu_data_ptr(getMluTensorImpl(grads[i]))};
          }
          tensors_ptr_list.emplace_back(std::move(tensors_ptr_array));
        }
        // Call the internal function
        bang_torch_fused_sgd_internal<
            CPPTypeToCNRTTypeValue_v<scalar_t>,
            depth>(
            tensors_ptr_list,
            tensor_sizes_list,
            lr_ptr,
            lr,
            weight_decay,
            momentum,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr,
            found_inf_ptr,
            stream,
            k_type,
            k_dim,
            nram_size);
      });
}

void bang__fused_sgd_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (!momentum_buffer_list.empty()) {
    TORCH_CHECK_GT(momentum, 0);
    TORCH_CHECK(torch_mlu::check_fast_path_restrictions(
        {params, grads, momentum_buffer_list}));
    _fused_sgd_mlu_internal_<3>(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr,
        nullptr,
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK_EQ(momentum, 0);
    TORCH_CHECK(torch_mlu::check_fast_path_restrictions({params, grads}));
    if (is_first_step) {
      TORCH_WARN_ONCE(
          "`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
    }
    _fused_sgd_mlu_internal_<2>(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr,
        nullptr,
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
  }
}

void bang__fused_sgd_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const at::Tensor& lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    bang__fused_sgd_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr.item<double>(),
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  if (!momentum_buffer_list.empty()) {
    TORCH_CHECK_GT(momentum, 0);
    TORCH_CHECK(torch_mlu::check_fast_path_restrictions(
        {params, grads, momentum_buffer_list}));
    if (grad_scale.has_value()) {
      TORCH_CHECK(
          grad_scale->device() == params[0].device(),
          "grad_scale must be on the same MLU device as the params");
    }
    if (found_inf.has_value()) {
      TORCH_CHECK(
          found_inf->device() == params[0].device(),
          "found_inf must be on the same MLU device as the params");
    }
    TORCH_CHECK(
        lr.device() == params[0].device(),
        "found_inf must be on the same MLU device as the params");
    _fused_sgd_mlu_internal_<3>(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        1.0f,
        lr.data_ptr<float>(),
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK_EQ(momentum, 0);
    TORCH_CHECK(torch_mlu::check_fast_path_restrictions({params, grads}));
    if (is_first_step) {
      TORCH_WARN_ONCE(
          "`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
    }
    if (grad_scale.has_value()) {
      TORCH_CHECK(
          grad_scale->device() == params[0].device(),
          "grad_scale must be on the same MLU device as the params");
    }
    if (found_inf.has_value()) {
      TORCH_CHECK(
          found_inf->device() == params[0].device(),
          "found_inf must be on the same MLU device as the params");
    }
    TORCH_CHECK(
        lr.device() == params[0].device(),
        "found_inf must be on the same MLU device as the params");
    _fused_sgd_mlu_internal_<2>(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        1.0f,
        lr.data_ptr<float>(),
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
  }
}

} // namespace ops
} // namespace torch_mlu
