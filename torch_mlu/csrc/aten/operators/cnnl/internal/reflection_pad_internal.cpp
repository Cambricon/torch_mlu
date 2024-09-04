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
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto padding_vec = padding.vec();
  int pad[4];
  for (int i = 0; i < padding_vec.size(); i++) {
    pad[i] = static_cast<int>(padding_vec[i]);
  }
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;

  // unsqueeze shape on first dim when input is 3D
  std::vector<int64_t> self_shape(4, 1);
  std::vector<int64_t> output_shape(4, 1);
  for (int64_t i = 0; i < self.dim(); ++i) {
    if (self.dim() == 3) {
      self_shape[i + 1] = self.size(i);
      output_shape[i + 1] = output.size(i);
    } else {
      self_shape[i] = self.size(i);
      output_shape[i] = output.size(i);
    }
  }

  if (self.dim() == 3) {
    auto self_stride = get_contiguous_strides(self_shape);
    auto output_stride = get_contiguous_strides(output_shape);
    descInput.set(self, self_shape, self_stride, CNNL_LAYOUT_NCHW);
    descOutput.set(output, output_shape, output_stride, CNNL_LAYOUT_NCHW);
  } else {
    auto suggest_layout = suggest_cnnl_layout(self);
    descInput.set(self, suggest_layout);
    descOutput.set(output, suggest_layout);
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
            descInput.desc(),
            input_ptr,
            pad,
            descOutput.desc(),
            output_ptr));
      });
}

void cnnl_reflection_pad1d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef padding) {
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // unsqueeze pad
  auto padding_vec = padding.vec();
  int pad[4];
  pad[0] = pad[1] = 0;
  for (int i = 0; i < padding_vec.size(); i++) {
    pad[i + 2] = static_cast<int>(padding_vec[i]);
  }

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;

  // unsqueeze shape on last dim and
  // unsqueeze shape on first dim when input is 2D
  std::vector<int64_t> self_shape(4, 1);
  std::vector<int64_t> output_shape(4, 1);
  for (int64_t i = 0; i < self.dim(); ++i) {
    if (self.dim() == 2) {
      self_shape[i + 1] = self.size(i);
      output_shape[i + 1] = output.size(i);
    } else {
      self_shape[i] = self.size(i);
      output_shape[i] = output.size(i);
    }
  }

  auto self_stride = get_contiguous_strides(self_shape);
  auto output_stride = get_contiguous_strides(output_shape);
  descInput.set(self, self_shape, self_stride, CNNL_LAYOUT_NCHW);
  descOutput.set(output, output_shape, output_stride, CNNL_LAYOUT_NCHW);

  auto input_ptr = mlu_data_ptr(input_impl);
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
            descInput.desc(),
            input_ptr,
            pad,
            descOutput.desc(),
            output_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
