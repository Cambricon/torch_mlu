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

void cnnl_reflection_pad2d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef padding) {
  c10::SmallVector<int, 4> pad(padding.size());
  for (int i = 0; i < padding.size(); i++) {
    pad[i] = static_cast<int>(padding[i]);
  }
  // get current handle
  auto handle = getCurrentHandle();

  // unsqueeze shape on first dim when input is 3D
  c10::SmallVector<int64_t, 4> self_shape(4, 1);
  c10::SmallVector<int64_t, 4> output_shape(4, 1);
  for (int64_t i = 0; i < self.dim(); ++i) {
    if (self.dim() == 3) {
      self_shape[i + 1] = self.size(i);
      output_shape[i + 1] = output.size(i);
    } else {
      self_shape[i] = self.size(i);
      output_shape[i] = output.size(i);
    }
  }
  tensorDescPtr_t descInput;
  tensorDescPtr_t descOutput;
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  if (self.dim() == 3) {
    auto self_stride = get_contiguous_strides(self_shape);
    auto output_stride = get_contiguous_strides(output_shape);
    cnnlDataType_t self_data_type = getCnnlType(input_impl);
    cnnlDataType_t output_data_type = getCnnlType(output_impl);
    descInput = getTensorDesc(
        self_shape, self_stride, self_data_type, CNNL_LAYOUT_NCHW);
    descOutput = getTensorDesc(
        output_shape, output_stride, output_data_type, CNNL_LAYOUT_NCHW);
  } else {
    auto suggest_layout = suggest_cnnl_layout(self);
    descInput = getTensorDesc(input_impl, suggest_layout);
    descOutput = getTensorDesc(output_impl, suggest_layout);
  }

  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "MLU reflection_pad2d",
      [&] {
        TORCH_CNNL_CHECK(cnnlReflectionPad2d(
            handle,
            descInput.get(),
            input_ptr,
            pad.data(),
            descOutput.get(),
            output_ptr));
      });
}

void cnnl_reflection_pad1d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef padding) {
  // Cnnl needs a 4 size pad. When processing pad1d, fill with 0
  c10::SmallVector<int, 4> pad{0, 0};
  for (const auto& val : padding) {
    pad.push_back(static_cast<int>(val));
  }

  // get current handle
  auto handle = getCurrentHandle();

  // unsqueeze shape on last dim and
  // unsqueeze shape on first dim when input is 2D
  c10::SmallVector<int64_t, 4> self_shape(4, 1);
  c10::SmallVector<int64_t, 4> output_shape(4, 1);
  for (int64_t i = 0; i < self.dim(); ++i) {
    if (self.dim() == 2) {
      self_shape[i + 1] = self.size(i);
      output_shape[i + 1] = output.size(i);
    } else {
      self_shape[i] = self.size(i);
      output_shape[i] = output.size(i);
    }
  }

  auto input_impl = getMluTensorImpl(self);
  auto self_stride = get_contiguous_strides(self_shape);
  cnnlDataType_t self_data_type = getCnnlType(input_impl);
  auto descInput =
      getTensorDesc(self_shape, self_stride, self_data_type, CNNL_LAYOUT_NCHW);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_stride = get_contiguous_strides(output_shape);
  cnnlDataType_t output_data_type = getCnnlType(output_impl);
  auto descOutput = getTensorDesc(
      output_shape, output_stride, output_data_type, CNNL_LAYOUT_NCHW);
  auto output_ptr = mlu_data_ptr(output_impl);

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Bool,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "MLU reflection_pad1d",
      [&] {
        TORCH_CNNL_CHECK(cnnlReflectionPad2d(
            handle,
            descInput.get(),
            input_ptr,
            pad.data(),
            descOutput.get(),
            output_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
