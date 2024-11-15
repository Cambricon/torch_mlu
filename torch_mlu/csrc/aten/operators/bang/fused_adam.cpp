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

#include <iostream>
#include "aten/operators/bang/common_utils.h"

namespace torch_mlu {
namespace ops {

bool bang_fused_adam(
    const at::Tensor& _dummy_overflow_buf,
    at::TensorList grads,
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    int64_t step,
    int64_t mode,
    int64_t bias_correction,
    double weight_decay) {
  double beta1_correction = 1.0f;
  double beta2_correction_sqrt = 1.0f;
  if (bias_correction == 1) {
    beta1_correction = 1 - std::pow(beta1, step);
    beta2_correction_sqrt = std::sqrt(1 - std::pow(beta2, step));
  }

  // epsion_correction = epsilon * (sqrt(1 - beta2 ^ t))
  double epsilon_correction = epsilon * beta2_correction_sqrt;
  // learning_rate_correction = -1.0*lr*sqrtf(bias_correction2)  /
  // bias_correction1
  double learning_rate_correction =
      -1.0 * learning_rate * beta2_correction_sqrt / beta1_correction;
  // weight_decay_correction = bias_correction1 / sqrtf(bias_correction2)) *
  // decay
  double weight_decay_correction =
      weight_decay * beta1_correction / beta2_correction_sqrt;
  double beta1_minus = 1 - beta1;
  double beta2_minus = 1 - beta2;

  // Add tensor size and device check
  const int tensor_num = grads.size();
  TORCH_CHECK(tensor_num > 0, "tensor num need be greater than zero.");
  auto ref_device = grads[0].device();
  TORCH_CHECK(
      ref_device.type() == at::kPrivateUse1, "expected input to be on mlu.");
  auto stream = getCurMLUStream();
  const int64_t device_index = ref_device.index();
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
      "bang_fused_adam",
      [&]() {
        static constexpr int depth = 4;
        // Used for contiguous tensors and hold tensors utils kernel launch.
        std::vector<std::array<at::Tensor, depth>> contiguous_tensors_list;
        // avoid call getMluTensorImpl in bangc utils code
        std::vector<std::array<void*, depth>> contiguous_ptr_list;
        std::vector<int64_t> tensor_sizes_list;
        tensor_sizes_list.reserve(tensor_num);
        contiguous_tensors_list.reserve(tensor_num);
        contiguous_ptr_list.reserve(tensor_num);
        for (int i = 0; i < tensor_num; ++i) {
          const at::Tensor& grad = grads[i];
          const int64_t num_elements = grad.numel();
          if (num_elements == 0) {
            CNLOG(INFO) << "Adam: Skip zero element tensor.";
            continue;
          }
          tensor_sizes_list.push_back(num_elements);
          const at::Tensor& exp_avg = exp_avgs[i];
          const at::Tensor& exp_avg_sq = exp_avg_sqs[i];
          const at::Tensor& param = params[i];
          check_device_and_numel(
              ref_device, num_elements, param, exp_avg, exp_avg_sq);
          // is_non_overllaping_and_dense need to support later.
          auto memory_format = param.suggest_memory_format();
          auto param_contiguous = cnnl_contiguous(param, memory_format);
          auto grad_contiguous = cnnl_contiguous(grad, memory_format);
          auto exp_avg_contiguous = cnnl_contiguous(exp_avg, memory_format);
          auto exp_avg_sq_contiguous =
              cnnl_contiguous(exp_avg_sq, memory_format);
          std::array<void*, depth> contiguous_ptr = {
              getMluTensorImpl(param_contiguous)->mlu_data_ptr(),
              getMluTensorImpl(grad_contiguous)->mlu_data_ptr(),
              getMluTensorImpl(exp_avg_contiguous)->mlu_data_ptr(),
              getMluTensorImpl(exp_avg_sq_contiguous)->mlu_data_ptr()

          };
          contiguous_ptr_list.emplace_back(std::move(contiguous_ptr));
          std::array<at::Tensor, depth> contiguous_tensors = {
              param_contiguous,
              grad_contiguous,
              exp_avg_contiguous,
              exp_avg_sq_contiguous};
          contiguous_tensors_list.emplace_back(std::move(contiguous_tensors));
        }
        // Call the internal function
        apex_fused_adam_internal<CPPTypeToCNRTTypeValue_v<scalar_t>, depth>(
            contiguous_ptr_list,
            tensor_sizes_list,
            beta1,
            beta1_minus,
            beta2,
            beta2_minus,
            epsilon_correction,
            learning_rate_correction,
            static_cast<internal::ADAM_MODE>(mode),
            weight_decay,
            weight_decay_correction,
            stream,
            k_type,
            k_dim,
            nram_size);
        int zero_count = 0;
        for (int i = 0; i < tensor_num; ++i) {
          if (grads[i].numel() == 0) {
            ++zero_count;
            continue;
          }
          const int index = i - zero_count;
          if (is_copy_necessary(params[i], contiguous_tensors_list[index][0])) {
            params[i].copy_(contiguous_tensors_list[index][0]);
          }
          if (is_copy_necessary(
                  exp_avgs[i], contiguous_tensors_list[index][2])) {
            exp_avgs[i].copy_(contiguous_tensors_list[index][2]);
          }
          if (is_copy_necessary(
                  exp_avg_sqs[i], contiguous_tensors_list[index][3])) {
            exp_avg_sqs[i].copy_(contiguous_tensors_list[index][3]);
          }
        }
      });
  return true;
}

} // namespace ops
} // namespace torch_mlu
