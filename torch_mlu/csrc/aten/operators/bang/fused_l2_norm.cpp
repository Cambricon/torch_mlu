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

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor> bang_fused_l2_norm(
    const at::Tensor& _dummy_overflow_buf,
    at::TensorList inputs,
    bool per_tensor) {
  auto stream = getCurMLUStream();
  auto tensor_num = inputs.size();
  std::tuple<at::Tensor, at::Tensor> outputs;
  outputs = std::make_tuple(
      at::empty(
          1,
          at::TensorOptions()
              .dtype(at::ScalarType::Float)
              .device(at::kPrivateUse1)),
      at::empty(
          tensor_num,
          at::TensorOptions()
              .dtype(at::ScalarType::Float)
              .device(at::kPrivateUse1)));

  cnrtDataType_V2_t cnrt_type =
      cnnlType2CnrtType_V2(getCnnlType(getMluTensorImpl(inputs[0])));

  AddressList tensors;
  SizeList sizes;
  int tensor_count = 0;

  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  uint32_t union_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.x = core_dim;
  k_dim.y = union_number;
  k_dim.z = 1;

  int taskdim = k_dim.x * k_dim.y * k_dim.z;
  at::Tensor output_buffer = at::zeros(
      taskdim,
      at::TensorOptions()
          .dtype(at::ScalarType::Float)
          .device(at::kPrivateUse1));
  at::Tensor output_buffer_per_tensor;
  if (per_tensor) {
    output_buffer_per_tensor = at::zeros(
        tensor_num * taskdim,
        at::TensorOptions()
            .dtype(at::ScalarType::Float)
            .device(at::kPrivateUse1));
  }

  TORCH_CHECK(
      _dummy_overflow_buf.scalar_type() == at::ScalarType::Int,
      "MultiTensorL2Norm: The data type of overflow must be int32")
  int32_t* overflow = static_cast<int32_t*>(
      mlu_data_ptr(getMluTensorImpl(_dummy_overflow_buf)));
  float* output_ptr =
      static_cast<float*>(mlu_data_ptr(getMluTensorImpl(std::get<0>(outputs))));
  float* output_buffer_ptr =
      static_cast<float*>(mlu_data_ptr(getMluTensorImpl(output_buffer)));
  float* output_per_tensor_ptr = nullptr;
  float* output_buffer_per_tensor_ptr = nullptr;
  if (per_tensor) {
    output_per_tensor_ptr = static_cast<float*>(
        mlu_data_ptr(getMluTensorImpl(std::get<1>(outputs))));
    output_buffer_per_tensor_ptr = static_cast<float*>(
        mlu_data_ptr(getMluTensorImpl(output_buffer_per_tensor)));
  }

  int tensor_offset = 0;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      inputs[0].scalar_type(),
      "bang_fused_l2_norm",
      [&]() {
        for (int tensor_id = 0; tensor_id < tensor_num; ++tensor_id) {
          auto input = inputs[tensor_id];
          int64_t num_elements = input.numel();
          auto memory_format = input.suggest_memory_format();
          auto input_contiguous = cnnl_contiguous(input, memory_format);

          if (num_elements == 0) {
            CNLOG(INFO) << "MultiTensorL2Norm: Skip zero element tensor.";
            continue;
          }

          tensors.addresses[tensor_count] =
              mlu_data_ptr(getMluTensorImpl(input_contiguous));
          sizes.sizes[tensor_count] = num_elements;

          ++tensor_count;
          if (tensor_count == MAX_TENSOR_NUM) {
            bang_fused_l2_norm_internal(
                tensors,
                sizes,
                output_buffer_ptr,
                output_buffer_per_tensor_ptr + tensor_offset * taskdim,
                tensor_count,
                per_tensor,
                overflow,
                k_dim,
                k_type,
                stream,
                cnrt_type,
                false);
            tensor_count = 0;
            tensor_offset = tensor_id + 1;
          }
        }

        if (tensor_count != 0) {
          bang_fused_l2_norm_internal(
              tensors,
              sizes,
              output_buffer_ptr,
              output_buffer_per_tensor_ptr + tensor_offset * taskdim,
              tensor_count,
              per_tensor,
              overflow,
              k_dim,
              k_type,
              stream,
              cnrt_type,
              false);
        }

        // see NOTE [taskDim assumption]
        bang_fused_l2_norm_clean_internal(
            output_ptr,
            output_per_tensor_ptr,
            output_buffer_ptr,
            output_buffer_per_tensor_ptr,
            per_tensor,
            tensor_num,
            overflow,
            k_dim,
            k_type,
            stream,
            false);
      });
  return outputs;
}

} // namespace ops
} // namespace torch_mlu
