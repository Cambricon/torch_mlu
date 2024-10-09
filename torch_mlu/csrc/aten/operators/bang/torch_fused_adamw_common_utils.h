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
#include "aten/operators/bang/common_utils.h"

namespace torch_mlu {
namespace ops {

template <internal::ADAM_MODE mode, bool is_amsgrad>
void _fused_adam_common_mlu_impl_(
    const at::TensorList& params,
    const at::TensorList& grads,
    const at::TensorList& exp_avgs,
    const at::TensorList& exp_avg_sqs,
    const at::TensorList& max_exp_avg_sqs,
    const at::TensorList& state_steps,
    const at::Tensor& lr,
    const double learning_rate,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double epsilon,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = lr.defined() ? lr.data_ptr<float>() : nullptr;
  double beta1_minus = 1 - beta1;
  double beta2_minus = 1 - beta2;
  const int tensor_num = grads.size();

  auto stream = getCurMLUStream();
  // compute kernel dim
  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  k_dim.x = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.y = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  k_dim.z = 1;
  // get grad and param tensor cnrt type
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      params[0].scalar_type(),
      "fused_adamw_mlu",
      [&]() {
        static constexpr int depth = is_amsgrad == true ? 5 : 4;
        // avoid call getMluTensorImpl in bangc utils code
        std::vector<std::array<void*, depth>> tensors_ptr_list;
        std::vector<void*> steps_ptr_list;
        std::vector<int64_t> tensor_sizes_list;
        tensors_ptr_list.reserve(tensor_num);
        steps_ptr_list.reserve(tensor_num);
        tensor_sizes_list.reserve(tensor_num);
        for (int i = 0; i < tensor_num; ++i) {
          const at::Tensor& grad = grads[i];
          const int num_elements = grad.numel();
          if (num_elements == 0) {
            continue;
          }
          tensor_sizes_list.push_back(num_elements);
          std::array<void*, depth> tensors_ptr_array;
          if constexpr (is_amsgrad == true) {
            tensors_ptr_array = {
                mlu_data_ptr(getMluTensorImpl(params[i])),
                mlu_data_ptr(getMluTensorImpl(grad)),
                mlu_data_ptr(getMluTensorImpl(exp_avgs[i])),
                mlu_data_ptr(getMluTensorImpl(exp_avg_sqs[i])),
                mlu_data_ptr(getMluTensorImpl(max_exp_avg_sqs[i]))};
          } else {
            tensors_ptr_array = {
                mlu_data_ptr(getMluTensorImpl(params[i])),
                mlu_data_ptr(getMluTensorImpl(grad)),
                mlu_data_ptr(getMluTensorImpl(exp_avgs[i])),
                mlu_data_ptr(getMluTensorImpl(exp_avg_sqs[i]))};
          }
          tensors_ptr_list.emplace_back(std::move(tensors_ptr_array));
          steps_ptr_list.emplace_back(
              mlu_data_ptr(getMluTensorImpl(state_steps[i])));
        }
        // Call the internal function
        bang_torch_fused_adamw_internal<
            CPPTypeToCNRTTypeValue_v<scalar_t>,
            depth>(
            tensors_ptr_list,
            tensor_sizes_list,
            steps_ptr_list,
            lr_ptr,
            learning_rate,
            beta1,
            beta1_minus,
            beta2,
            beta2_minus,
            weight_decay,
            epsilon,
            maximize,
            is_amsgrad,
            grad_scale_ptr,
            found_inf_ptr,
            mode,
            stream,
            k_type,
            k_dim);
      });
}

} // namespace ops
} // namespace torch_mlu
