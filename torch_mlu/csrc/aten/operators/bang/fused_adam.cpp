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

#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"
#include "aten/utils/utils.h"

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
  auto stream = getCurMLUStream();
  auto tensor_num = grads.size();
  float beta1_correction_recip = 1.0f;
  float beta2_correction_recip = 1.0f;
  if (bias_correction == 1) {
    beta1_correction_recip = 1 / (1 - std::pow(beta1, step));
    beta2_correction_recip = 1 / (1 - std::pow(beta2, step));
  }

  // epsion_correction = epsilon * (sqrt(1 - beta2 ^ t))
  float epsilon_correction = epsilon / std::sqrt(beta2_correction_recip);
  float learning_rate_correction = learning_rate * beta1_correction_recip /
      std::sqrt(beta2_correction_recip);
  float weight_decay_correction =
      weight_decay / beta1_correction_recip * std::sqrt(beta2_correction_recip);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      params[0].scalar_type(),
      "bang_fused_adam",
      [&]() {
        cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
        cnrtDim3_t k_dim;
        uint32_t union_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
        uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
        k_dim.x = core_dim;
        k_dim.y = union_number;
        k_dim.z = 1;
        cnrtDataType_V2_t cnrt_type =
            cnnlType2CnrtType_V2(getCnnlType(getMluTensorImpl(grads[0])));

        AddressList g, m, v, p;
        SizeList sizes;
        int tensor_count = 0;
        std::vector<std::vector<at::Tensor>> contiguous_tensors_list;
        for (int64_t i = 0; i < tensor_num; ++i) {
          at::Tensor grad = grads[i];
          at::Tensor exp_avg = exp_avgs[i];
          at::Tensor exp_avg_sq = exp_avg_sqs[i];
          at::Tensor param = params[i];
          int64_t num_elements = grad.numel();
          std::vector<at::Tensor> contiguous_tensors;

          auto memory_format = param.suggest_memory_format();
          auto grad_contiguous = cnnl_contiguous(grad, memory_format);
          auto exp_avg_contiguous = cnnl_contiguous(exp_avg);
          auto exp_avg_sq_contiguous = cnnl_contiguous(exp_avg_sq);
          auto param_contiguous = cnnl_contiguous(param, memory_format);

          contiguous_tensors.push_back(exp_avg_contiguous);
          contiguous_tensors.push_back(exp_avg_sq_contiguous);
          contiguous_tensors.push_back(param_contiguous);
          contiguous_tensors_list.push_back(contiguous_tensors);

          auto grad_ptr = getMluTensorImpl(grad_contiguous)->mlu_data_ptr();
          auto exp_avg_ptr =
              getMluTensorImpl(exp_avg_contiguous)->mlu_data_ptr();
          auto exp_avg_sq_ptr =
              getMluTensorImpl(exp_avg_sq_contiguous)->mlu_data_ptr();
          auto param_ptr = getMluTensorImpl(param_contiguous)->mlu_data_ptr();

          if (num_elements == 0) {
            CNLOG(INFO) << "Adam: Skip zero element tensor.";
            continue;
          }

          g.addresses[tensor_count] = grad_ptr;
          m.addresses[tensor_count] = exp_avg_ptr;
          v.addresses[tensor_count] = exp_avg_sq_ptr;
          p.addresses[tensor_count] = param_ptr;
          sizes.sizes[tensor_count] = num_elements;

          ++tensor_count;
          if (tensor_count == MAX_TENSOR_NUM) {
            bang_fused_adam_internal(
                g,
                m,
                v,
                p,
                sizes,
                tensor_count,
                beta1,
                beta2,
                beta1_correction_recip,
                beta2_correction_recip,
                epsilon,
                epsilon_correction,
                learning_rate,
                learning_rate_correction,
                mode,
                weight_decay,
                weight_decay_correction,
                k_dim,
                k_type,
                stream,
                cnrt_type);
            tensor_count = 0;
          }
        }
        if (tensor_count != 0) {
          bang_fused_adam_internal(
              g,
              m,
              v,
              p,
              sizes,
              tensor_count,
              beta1,
              beta2,
              beta1_correction_recip,
              beta2_correction_recip,
              epsilon,
              epsilon_correction,
              learning_rate,
              learning_rate_correction,
              mode,
              weight_decay,
              weight_decay_correction,
              k_dim,
              k_type,
              stream,
              cnrt_type);
        }

        for (int64_t i = 0; i < tensor_num; ++i) {
          if (is_copy_necessary(exp_avgs[i], contiguous_tensors_list[i][0])) {
            exp_avgs[i].copy_(contiguous_tensors_list[i][0]);
          }
          if (is_copy_necessary(
                  exp_avg_sqs[i], contiguous_tensors_list[i][1])) {
            exp_avg_sqs[i].copy_(contiguous_tensors_list[i][1]);
          }
          if (is_copy_necessary(params[i], contiguous_tensors_list[i][2])) {
            params[i].copy_(contiguous_tensors_list[i][2]);
          }
        }
      });

  return true;
}

} // namespace ops
} // namespace torch_mlu
