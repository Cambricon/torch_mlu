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
#include "aten/operators/bang/common_utils.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor> bang_fused_l2_norm_common(
    const at::Tensor& _dummy_overflow_buf,
    at::TensorList& inputs,
    c10::optional<bool>& per_tensor_python,
    const bool& is_amp) {
  // add high sqrt percision control.
  static bool high_sqrt_precision = is_high_sqrt_precision();
  bool per_tensor =
      per_tensor_python.has_value() ? per_tensor_python.value() : false;
  auto tensor_num = inputs.size();
  TORCH_CHECK(tensor_num > 0, "tensor num need be greater than zero.");

  // Add tensor size and device check
  auto ref_device = inputs[0].device();
  TORCH_CHECK(
      ref_device.type() == at::kPrivateUse1, "expected input to be on mlu.");
  auto stream = getCurMLUStream();
  const int64_t device_index = ref_device.index();
  cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;
  cnrtDim3_t k_dim;
  k_dim.x = torch_mlu::getDeviceProperties(device_index)->core_num_per_cluster;
  k_dim.y = torch_mlu::getDeviceProperties(device_index)->cluster_count;
  k_dim.z = 1;
  const int nram_size = torch_mlu::getDeviceProperties(device_index)->nram_size;

  int taskdim = k_dim.x * k_dim.y * k_dim.z;
  at::Tensor output = at::empty(
      1,
      at::TensorOptions()
          .dtype(at::ScalarType::Float)
          .device(at::kPrivateUse1));
  //  Using buffer to store intermediate result. And buffer must be init to
  //  zero.
  //  1) For total norm of all input tensors, each ipu will get a intermediate
  //     result, which is the sum result of all computations on this core.
  //  2) For per tensor norm of each input tensors, each ipu will store
  //     accumulate result of each tensor to corresponding position.
  //     After this, through sram to reduce sum all intermediate result, then
  //     store to output per tensor buffer.
  at::Tensor output_buffer = at::zeros(
      taskdim, at::TensorOptions().dtype(at::kFloat).device(at::kPrivateUse1));
  at::Tensor output_per_tensor;
  at::Tensor output_buffer_per_tensor;
  if (per_tensor) {
    output_per_tensor = std::move(at::empty(
        tensor_num,
        at::TensorOptions()
            .dtype(at::ScalarType::Float)
            .device(at::kPrivateUse1)));
    // buffer for cluster intermediate result. k_dim.y mean num of buffer, and
    // each buffer size is tensor_num * sizeof(float)
    output_buffer_per_tensor = at::zeros(
        tensor_num * k_dim.y,
        at::TensorOptions().dtype(at::kFloat).device(at::kPrivateUse1));
  }

  TORCH_CHECK(
      _dummy_overflow_buf.scalar_type() == at::ScalarType::Int,
      "MultiTensorL2Norm: The data type of overflow must be int32")
  int32_t* overflow = static_cast<int32_t*>(
      getMluTensorImpl(_dummy_overflow_buf)->mlu_data_ptr());
  float* output_ptr =
      static_cast<float*>(getMluTensorImpl(output)->mlu_data_ptr());
  float* output_buffer_ptr =
      static_cast<float*>(getMluTensorImpl(output_buffer)->mlu_data_ptr());
  float* output_per_tensor_ptr = nullptr;
  float* output_buffer_per_tensor_ptr = nullptr;
  if (per_tensor) {
    output_per_tensor_ptr = static_cast<float*>(
        getMluTensorImpl(output_per_tensor)->mlu_data_ptr());
    output_buffer_per_tensor_ptr = static_cast<float*>(
        getMluTensorImpl(output_buffer_per_tensor)->mlu_data_ptr());
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      inputs[0].scalar_type(),
      "bang_fused_l2_norm",
      [&]() {
        constexpr int depth = 1;
        std::vector<at::Tensor> contiguous_tensor_list;
        std::vector<std::array<void*, depth>> tensor_ptr_list;
        std::vector<int64_t> tensor_size_list;
        std::vector<int> tensor_index_list;
        contiguous_tensor_list.reserve(tensor_num);
        tensor_ptr_list.reserve(tensor_num);
        tensor_size_list.reserve(tensor_num);
        tensor_index_list.reserve(tensor_num);
        for (int index = 0; index < tensor_num; ++index) {
          auto input = inputs[index];
          int64_t num_elements = input.numel();
          if (num_elements == 0) {
            CNLOG(INFO) << "MultiTensorL2Norm: Skip zero element tensor.";
            continue;
          }
          TORCH_CHECK(input.device() == ref_device, "Device need be same.");
          auto memory_format = input.suggest_memory_format();
          auto input_contiguous = cnnl_contiguous(input, memory_format);
          std::array<void*, depth> ptr_array = {
              getMluTensorImpl(input_contiguous)->mlu_data_ptr()};
          tensor_ptr_list.emplace_back(std::move(ptr_array));
          tensor_size_list.emplace_back(num_elements);
          contiguous_tensor_list.emplace_back(std::move(input_contiguous));
          tensor_index_list.emplace_back(index);
        }
        if (tensor_ptr_list.size() == 0) {
          // GPU malloc and init device memory to zero.
          // So need to fill zero.
          output.fill_(0.0f);
          if (per_tensor)
            output_per_tensor.fill_(0.0f);
        } else {
          bang_fused_l2_norm_internal<
              CPPTypeToCNRTTypeValue_v<scalar_t>,
              depth>(
              tensor_ptr_list,
              tensor_size_list,
              tensor_index_list,
              output_buffer_ptr,
              tensor_num,
              output_buffer_per_tensor_ptr,
              per_tensor,
              overflow,
              k_dim,
              k_type,
              nram_size,
              stream,
              is_amp);
          // using block task to reduce cluster data to
          // output and output_per_tensor
          const int cluster_num = k_dim.y;
          k_type = cnrtFuncTypeBlock;
          k_dim.x = 1;
          k_dim.y = 1;
          bang_fused_l2_norm_clean_internal(
              taskdim,
              output_ptr,
              output_buffer_ptr,
              cluster_num,
              tensor_num,
              output_per_tensor_ptr,
              output_buffer_per_tensor_ptr,
              per_tensor,
              overflow,
              k_dim,
              k_type,
              stream,
              is_amp,
              high_sqrt_precision);
        }
      });
  return {output, output_per_tensor};
}

std::tuple<at::Tensor, at::Tensor> bang_fused_l2_norm(
    const at::Tensor& _dummy_overflow_buf,
    at::TensorList inputs,
    c10::optional<bool> per_tensor_python) {
  return bang_fused_l2_norm_common(
      _dummy_overflow_buf, inputs, per_tensor_python, false);
}

// reference to multi_tensor_l2norm_kernel_mp
std::tuple<at::Tensor, at::Tensor> bang_fused_l2_norm_amp(
    const at::Tensor& _dummy_overflow_buf,
    at::TensorList inputs,
    c10::optional<bool> per_tensor_python) {
  return bang_fused_l2_norm_common(
      _dummy_overflow_buf, inputs, per_tensor_python, true);
}

} // namespace ops
} // namespace torch_mlu
