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

static void cnnl_logic_internal_impl(
    at::Tensor& output,
    tensorDescPtr_t input_desc,
    void* input_ptr,
    tensorDescPtr_t other_desc,
    void* other_ptr,
    cnnlLogicOp_t logic_type) {
  auto handle = getCurrentHandle();
  auto output_impl = getMluTensorImpl(output);

  // get cnnl descriptor
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto output_ptr = mlu_data_ptr(output_impl);

  // compute size of workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetLogicOpWorkspaceSize(
      handle,
      input_desc.get(),
      other_desc.get(),
      output_desc.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlLogicOp(
      handle,
      logic_type,
      input_desc.get(),
      input_ptr,
      other_desc.get(),
      other_ptr,
      workspace_ptr.get(),
      workspace_size,
      output_desc.get(),
      output_ptr));
}

void cnnl_logic_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& other,
    cnnlLogicOp_t logic_type,
    const at::ScalarType& compute_dtype) {
  TORCH_CHECK(
      input.dim() <= CNNL_MAX_DIM_SIZE && other.dim() <= CNNL_MAX_DIM_SIZE,
      "all input tensors dimension should less than ",
      CNNL_MAX_DIM_SIZE,
      ", but now input dimension is ",
      input.dim(),
      " other dimension is ",
      other.dim());
  auto handle = getCurrentHandle();
  // Input and other need be some dtype, if one of them is a CPU tensor.
  // And already checked datatype convert overflow
  // in pytorch wrapped_scalar_tensor_and_check_convert function.
  auto compute_dtype_ = compute_dtype == at::ScalarType::Undefined
      ? output.scalar_type()
      : compute_dtype;
  if (other.is_cpu() && other.numel() == 1) {
    AT_DISPATCH_ALL_TYPES_AND3(
        at::kHalf,
        at::kBFloat16,
        at::kBool,
        compute_dtype_,
        "logic_internal with other scalar",
        [&]() {
          using mlu_scalar_t = torch_mlu::Convert64BitTo32Bit_t<scalar_t>;
          auto scalar_type = other.scalar_type();
          mlu_scalar_t value = c10::fetch_and_cast<mlu_scalar_t>(
              scalar_type, other.const_data_ptr());
          auto input_impl = getMluTensorImpl(input);
          auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
          auto input_ptr = mlu_data_ptr(input_impl);
          auto other_desc = getCpuTensorDesc(
              torch_mlu::getCnnlDataType(compute_dtype_),
              CNNL_POINTER_MODE_HOST);
          cnnl_logic_internal_impl(
              output,
              std::move(input_desc),
              input_ptr,
              std::move(other_desc),
              &value,
              logic_type);
        });
  } else if (input.is_cpu() && input.numel() == 1) {
    AT_DISPATCH_ALL_TYPES_AND3(
        at::kHalf,
        at::kBFloat16,
        at::kBool,
        compute_dtype_,
        "logic_internal with input scalar",
        [&]() {
          using mlu_scalar_t = torch_mlu::Convert64BitTo32Bit_t<scalar_t>;
          auto scalar_type = input.scalar_type();
          mlu_scalar_t value = c10::fetch_and_cast<mlu_scalar_t>(
              scalar_type, input.const_data_ptr());
          auto other_impl = getMluTensorImpl(other);
          auto other_ptr = mlu_data_ptr(other_impl);
          auto other_desc = getTensorDesc(other_impl, CNNL_LAYOUT_ARRAY);
          auto input_desc = getCpuTensorDesc(
              torch_mlu::getCnnlDataType(compute_dtype_),
              CNNL_POINTER_MODE_HOST);
          cnnl_logic_internal_impl(
              output,
              std::move(input_desc),
              &value,
              std::move(other_desc),
              other_ptr,
              logic_type);
        });
  } else {
    auto input_impl = getMluTensorImpl(input);
    auto input_ptr = mlu_data_ptr(input_impl);
    auto other_impl = getMluTensorImpl(other);
    auto other_ptr = mlu_data_ptr(other_impl);
    auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
    auto other_desc = getTensorDesc(other_impl, CNNL_LAYOUT_ARRAY);
    cnnl_logic_internal_impl(
        output,
        std::move(input_desc),
        input_ptr,
        std::move(other_desc),
        other_ptr,
        logic_type);
  }
}

} // namespace ops
} // namespace torch_mlu
