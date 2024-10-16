/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
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

void cnnl_reflection_pad2d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef padding) {
  c10::SmallVector<int, 4> pad(padding.size());
  for (int i = 0; i < padding.size(); i++) {
    pad[i] = static_cast<int>(padding[i]);
  }

  // get current handle
  auto handle = getCurrentHandle();

  // unsqueeze shape on first dim when input is 3D
  c10::SmallVector<int64_t, 4> grad_input_shape(4, 1);
  c10::SmallVector<int64_t, 4> grad_output_shape(4, 1);
  for (int64_t i = 0; i < grad_input.dim(); ++i) {
    if (grad_input.dim() == 3) {
      grad_input_shape[i + 1] = grad_input.size(i);
      grad_output_shape[i + 1] = grad_output.size(i);
    } else {
      grad_input_shape[i] = grad_input.size(i);
      grad_output_shape[i] = grad_output.size(i);
    }
  }
  auto grad_input_stride = get_contiguous_strides(grad_input_shape);
  auto grad_input_impl = getMluTensorImpl(grad_input);
  cnnlDataType_t grad_input_dtype = getCnnlType(grad_input_impl);
  auto desc_grad_input = getTensorDesc(
      grad_input_shape, grad_input_stride, grad_input_dtype, CNNL_LAYOUT_NCHW);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);

  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_output_stride = get_contiguous_strides(grad_output_shape);
  cnnlDataType_t grad_output_dtype = getCnnlType(grad_output_impl);
  auto desc_grad_output = getTensorDesc(
      grad_output_shape,
      grad_output_stride,
      grad_output_dtype,
      CNNL_LAYOUT_NCHW);
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_input.scalar_type(),
      "MLU reflection_pad2d_backward",
      [&] {
        TORCH_CNNL_CHECK(cnnlReflectionPadBackward(
            handle,
            desc_grad_output.get(),
            grad_output_ptr,
            pad.data(),
            desc_grad_input.get(),
            grad_input_ptr));
      });
}

void cnnl_reflection_pad1d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef padding) {
  // Cnnl needs a 4 size pad. When processing pad1d, fill with 0
  c10::SmallVector<int, 4> pad{0, 0};
  for (const auto& val : padding) {
    pad.push_back(static_cast<int>(val));
  }

  // get current handle
  auto handle = getCurrentHandle();

  // unsqueeze shape on last dim and
  // unsqueeze shape on first dim when grad_input is 2D
  c10::SmallVector<int64_t, 4> grad_input_shape(4, 1);
  c10::SmallVector<int64_t, 4> grad_output_shape(4, 1);
  for (int64_t i = 0; i < grad_input.dim(); ++i) {
    if (grad_input.dim() == 2) {
      grad_input_shape[i + 1] = grad_input.size(i);
      grad_output_shape[i + 1] = grad_output.size(i);
    } else {
      grad_input_shape[i] = grad_input.size(i);
      grad_output_shape[i] = grad_output.size(i);
    }
  }
  auto grad_input_impl = getMluTensorImpl(grad_input);
  cnnlDataType_t grad_input_data_type = getCnnlType(grad_input_impl);
  auto grad_input_stride = get_contiguous_strides(grad_input_shape);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);
  auto desc_grad_input = getTensorDesc(
      grad_input_shape,
      grad_input_stride,
      grad_input_data_type,
      CNNL_LAYOUT_NCHW);

  auto grad_output_impl = getMluTensorImpl(grad_output);
  cnnlDataType_t grad_output_data_type = getCnnlType(grad_output_impl);
  auto grad_output_stride = get_contiguous_strides(grad_output_shape);
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);
  auto desc_grad_output = getTensorDesc(
      grad_output_shape,
      grad_output_stride,
      grad_output_data_type,
      CNNL_LAYOUT_NCHW);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_input.scalar_type(),
      "MLU reflection_pad1d_backward",
      [&] {
        TORCH_CNNL_CHECK(cnnlReflectionPadBackward(
            handle,
            desc_grad_output.get(),
            grad_output_ptr,
            pad.data(),
            desc_grad_input.get(),
            grad_input_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
