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

#include "aten/operators/bang/common_utils.h"

namespace torch_mlu {
namespace ops {

#define DISPATCH_FLOAT_HALF_AND_BFLOAT(TYPE, LEVEL, NAME, ...)        \
  switch (TYPE) {                                                     \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t_##LEVEL = at::Half;                              \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::BFloat16: {                                  \
      using scalar_t_##LEVEL = at::BFloat16;                          \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

bool multi_tensor_fused_adam_impl(
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList grads,
    at::TensorList params_out,
    const at::Tensor& grad_scale,
    float lr,
    float* lr_ptr,
    float beta1,
    float beta2,
    float eps,
    int step,
    int* step_ptr,
    int mode,
    int bias_correction,
    float weight_decay) {
  using namespace at;

  size_t tensor_num = params.size();
  TORCH_CHECK(tensor_num > 0, "tensor num must be greater than zero.");

  auto mlu_stream = getCurMLUStream();
  auto ref_device = params[0].device();
  TORCH_CHECK(
      ref_device.type() == at::kPrivateUse1,
      "expect device type of input is MLU.");
  const int64_t device_index = ref_device.index();
  // kernel task dim.
  cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;
  cnrtDim3_t k_dim;
  k_dim.x = torch_mlu::getDeviceProperties(device_index)->core_num_per_cluster;
  k_dim.y = torch_mlu::getDeviceProperties(device_index)->cluster_count;
  k_dim.z = 1;

  const int nram_size = torch_mlu::getDeviceProperties(device_index)->nram_size;

  auto p_in_type = params[0].scalar_type();
  auto g_type = grads[0].scalar_type();
  auto p_out_type = params_out[0].scalar_type();

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      p_in_type,
      0,
      "multi_tensor_fused_adam_impl",
      DISPATCH_FLOAT_HALF_AND_BFLOAT(
          g_type,
          1,
          "multi_tensor_fused_adam_impl",
          DISPATCH_FLOAT_HALF_AND_BFLOAT(
              p_out_type,
              2,
              "multi_tensor_fused_adam_impl",
              static constexpr int depth = 5;
              std::vector<std::array<void*, depth>> data_ptr_list;
              std::vector<int64_t> tensor_sizes_list;
              data_ptr_list.reserve(tensor_num);
              for (int i = 0; i < tensor_num; i++) {
                const int64_t num_element = grads[i].numel();
                if (num_element == 0) {
                  CNLOG(INFO)
                      << "multi_tensor_fused_adam_impl: grad is zero element tensor.";
                  continue;
                }
                tensor_sizes_list.push_back(num_element);
                const at::Tensor& param = params[i];
                const at::Tensor& exp_avg = exp_avgs[i];
                const at::Tensor& exp_avg_sq = exp_avg_sqs[i];
                const at::Tensor& grad = grads[i];
                const at::Tensor& param_out = params_out[i];
                check_device_and_numel(
                    ref_device,
                    num_element,
                    param,
                    exp_avg,
                    exp_avg_sq,
                    param_out);
                check_contiguous(param, exp_avg, exp_avg_sq, grad, param_out);
                // param_ptr, exp_avg_ptr, exp_avg_sq_ptr, grad_ptr,
                // param_out_ptr
                std::array<void*, depth> tensors_data_ptr = {
                    mlu_data_ptr(getMluTensorImpl(param)),
                    mlu_data_ptr(getMluTensorImpl(exp_avg)),
                    mlu_data_ptr(getMluTensorImpl(exp_avg_sq)),
                    mlu_data_ptr(getMluTensorImpl(grad)),
                    mlu_data_ptr(getMluTensorImpl(param_out))};
                data_ptr_list.emplace_back(std::move(tensors_data_ptr));
              } if ((lr_ptr && step_ptr) || !(lr_ptr || step_ptr)) {
                multi_tensor_fused_adam_internal<
                    CPPTypeToCNRTTypeValue_v<scalar_t_0>,
                    CPPTypeToCNRTTypeValue_v<scalar_t_1>,
                    CPPTypeToCNRTTypeValue_v<scalar_t_2>,
                    depth>(
                    data_ptr_list,
                    grad_scale.data_ptr<float>(),
                    tensor_sizes_list,
                    beta1,
                    beta2,
                    step,
                    step_ptr,
                    static_cast<internal::ADAM_MODE>(mode),
                    eps,
                    bias_correction,
                    lr,
                    lr_ptr,
                    weight_decay,
                    mlu_stream,
                    k_type,
                    k_dim,
                    nram_size);
              } else {
                AT_ERROR(
                    "Both lr_ptr and step_ptr are nullptr or neither, but now lr_ptr is ",
                    lr_ptr,
                    " step_ptr is ",
                    step_ptr);
              })));
  return true;
}

bool bang_multi_tensor_fused_adam(
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList grads,
    at::TensorList params_out,
    const at::Tensor& grad_scale,
    double lr,
    double beta1,
    double beta2,
    double eps,
    int64_t step,
    int64_t mode,
    int64_t bias_correction,
    double weight_decay) {
  return multi_tensor_fused_adam_impl(
      params,
      exp_avgs,
      exp_avg_sqs,
      grads,
      params_out,
      grad_scale,
      static_cast<float>(lr),
      nullptr, /*lr_ptr placeholder*/
      static_cast<float>(beta1),
      static_cast<float>(beta2),
      static_cast<float>(eps),
      static_cast<int>(step),
      nullptr, /*step_ptr placeholder*/
      static_cast<int>(mode),
      static_cast<int>(bias_correction),
      static_cast<float>(weight_decay));
}

bool bang_multi_tensor_fused_adam_capturable(
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList grads,
    at::TensorList params_out,
    const at::Tensor& grad_scale,
    const at::Tensor& lr,
    double beta1,
    double beta2,
    double eps,
    const at::Tensor& step,
    int64_t mode,
    int64_t bias_correction,
    double weight_decay) {
  return multi_tensor_fused_adam_impl(
      params,
      exp_avgs,
      exp_avg_sqs,
      grads,
      params_out,
      grad_scale,
      0, /*lr placeholder*/
      lr.data_ptr<float>(),
      static_cast<float>(beta1),
      static_cast<float>(beta2),
      static_cast<float>(eps),
      0, /*step placeholder*/
      step.data_ptr<int>(),
      static_cast<int>(mode),
      static_cast<int>(bias_correction),
      static_cast<float>(weight_decay));
}

bool bang_multi_tensor_fused_adam_remainder(
    at::TensorList params_in,
    at::TensorList params_rem,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList grads,
    at::TensorList params_out,
    const at::Tensor& grad_scale,
    double lr,
    double beta1,
    double beta2,
    double eps,
    int64_t step,
    int64_t mode,
    int64_t bias_correction,
    double weight_decay) {
  using namespace at;

  size_t tensor_num = params_in.size();
  TORCH_CHECK(tensor_num > 0, "tensor num must be greater than zero.");

  auto mlu_stream = getCurMLUStream();
  auto ref_device = params_in[0].device();
  TORCH_CHECK(
      ref_device.type() == at::kPrivateUse1,
      "expect device type of input is MLU.");
  const int64_t device_index = ref_device.index();
  // kernel task dim.
  cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;
  cnrtDim3_t k_dim;
  k_dim.x = torch_mlu::getDeviceProperties(device_index)->core_num_per_cluster;
  k_dim.y = torch_mlu::getDeviceProperties(device_index)->cluster_count;
  k_dim.z = 1;

  const int nram_size = torch_mlu::getDeviceProperties(device_index)->nram_size;

  auto p_in_type = params_in[0].scalar_type();
  auto p_rem_type = params_rem[0].scalar_type();
  auto g_type = grads[0].scalar_type();
  auto p_out_type = params_out[0].scalar_type();
  TORCH_CHECK(
      p_in_type == at::ScalarType::BFloat16,
      "multi_tensor_fused_adam_remainder: params_in dtype must be BFloat16.");
  TORCH_CHECK(
      p_rem_type == at::ScalarType::Short,
      "multi_tensor_fused_adam_remainder: params_rem dtype must be Short.");
  TORCH_CHECK(
      p_out_type == at::ScalarType::BFloat16,
      "multi_tensor_fused_adam_remainder: params_out dtype must be BFloat16.");

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      g_type,
      0,
      "multi_tensor_fused_adam_remainder",
      static constexpr int depth = 6;
      std::vector<std::array<void*, depth>> data_ptr_list;
      std::vector<int64_t> tensor_sizes_list;
      data_ptr_list.reserve(tensor_num);
      for (int i = 0; i < tensor_num; i++) {
        const int64_t num_element = grads[i].numel();
        if (num_element == 0) {
          CNLOG(INFO)
              << "multi_tensor_fused_adam_impl: grad is zero element tensor.";
          continue;
        }
        tensor_sizes_list.push_back(num_element);
        const at::Tensor& param_in = params_in[i];
        const at::Tensor& param_rem = params_rem[i];
        const at::Tensor& exp_avg = exp_avgs[i];
        const at::Tensor& exp_avg_sq = exp_avg_sqs[i];
        const at::Tensor& grad = grads[i];
        const at::Tensor& param_out = params_out[i];
        check_device_and_numel(
            ref_device,
            num_element,
            param_in,
            param_rem,
            exp_avg,
            exp_avg_sq,
            param_out);
        check_contiguous(param_in, param_rem, exp_avg, exp_avg_sq, grad, param_out);
        // param_in_ptr, param_rem_ptr, exp_avg_ptr, exp_avg_sq_ptr, grad_ptr, param_out_ptr
        std::array<void*, depth> tensors_data_ptr = {
            mlu_data_ptr(getMluTensorImpl(param_in)),
            mlu_data_ptr(getMluTensorImpl(param_rem)),
            mlu_data_ptr(getMluTensorImpl(exp_avg)),
            mlu_data_ptr(getMluTensorImpl(exp_avg_sq)),
            mlu_data_ptr(getMluTensorImpl(grad)),
            mlu_data_ptr(getMluTensorImpl(param_out))};
        data_ptr_list.emplace_back(std::move(tensors_data_ptr));
      }
      multi_tensor_fused_adam_remainders_internal<
          CPPTypeToCNRTTypeValue_v<scalar_t_0>,
          depth>(
          data_ptr_list,
          grad_scale.data_ptr<float>(),
          tensor_sizes_list,
          beta1,
          beta2,
          step,
          static_cast<internal::ADAM_MODE>(mode),
          eps,
          bias_correction,
          lr,
          weight_decay,
          mlu_stream,
          k_type,
          k_dim,
          nram_size);
      );
  return true;
}

#undef DISPATCH_FLOAT_HALF_AND_BFLOAT

} // namespace ops
} // namespace torch_mlu
