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
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {
// The cnnl_masked_fill_internal function will accept two values (both tensor
// and scalar) and will use one of them depends on mask_op
at::Tensor& cnnl_masked_fill_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& mask,
    const at::Tensor& value_tensor) {
  auto input_impl = getMluTensorImpl(input);
  auto mask_impl = getMluTensorImpl(mask);
  auto output_impl = getMluTensorImpl(output);

  // cnnlMasked_v4 only supports CNNL_LAYOUT_ARRAY layout
  auto desc_input = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto desc_mask = getTensorDesc(mask_impl, CNNL_LAYOUT_ARRAY);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

  auto input_ptr = mlu_data_ptr(input_impl);
  auto mask_ptr = mlu_data_ptr(mask_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get handle
  auto handle = getCurrentHandle();

  // get workspace size
  size_t workspace_size = 0;
  bool is_scalar_value = isCpuScalar(value_tensor);
  // create one more tensor desc when fill host mode,
  // but a lot of duplicate code be deleted.
  // Using new a CnnlTensorDescriptor
  // will fix this. But can we using this in internal.cpp?
  tensorDescPtr_t desc_value;
  void* device_value_ptr = nullptr;
  void* scalar_value_ptr = nullptr;
  auto mask_mode = CNNL_MASKED_FILL_HOST;
  if (!is_scalar_value) {
    auto* value_impl = getMluTensorImpl(value_tensor);
    desc_value = getTensorDesc(value_impl, CNNL_LAYOUT_ARRAY);
    device_value_ptr = mlu_data_ptr(value_impl);
    mask_mode = CNNL_MASKED_FILL;
  }
  TORCH_CNNL_CHECK(cnnlGetMaskedWorkspaceSize(
      handle,
      mask_mode,
      desc_input.get(),
      desc_mask.get(),
      desc_value.get(),
      desc_output.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // cnnlMasked
  // AT_DISPATCH_ALL_TYPES_AND3 support kByte, but cnnlMasked_v4 not support
  // uint8 in masked fill mode, so we need to check input type.
  TORCH_CHECK(
      input.scalar_type() != at::kByte,
      "input type is not support uint8 in cnnl_masked_fill_internal");
  AT_DISPATCH_ALL_TYPES_AND3(
      at::kBool,
      at::kBFloat16,
      at::kHalf,
      input.scalar_type(),
      "masked_fill_internal",
      [&] {
        // see Note: [Convert64BitTo32Bit] in accumulate_type.h
        // for more details
        using catch_scalar_t = torch_mlu::Convert64BitTo32Bit_t<scalar_t>;
        catch_scalar_t scalar_value;
        if (is_scalar_value) {
          scalar_value = value_tensor.item().to<catch_scalar_t>();
          scalar_value_ptr = (void*)(&scalar_value);
        }
        TORCH_CNNL_CHECK(cnnlMasked_v4(
            handle,
            mask_mode,
            desc_input.get(),
            input_ptr,
            desc_mask.get(),
            mask_ptr,
            desc_value.get(),
            device_value_ptr,
            scalar_value_ptr,
            workspace_ptr.get(),
            workspace_size,
            desc_output.get(),
            output_ptr,
            nullptr));
      });
  return output;
}

} // namespace ops
} // namespace torch_mlu
