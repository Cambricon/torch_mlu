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
#include "c10/core/ScalarType.h"

namespace torch_mlu {
namespace ops {

void cnnl_adaptive_avg_pool_internal(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  auto C = input.size(1);
  int64_t ndim = input.dim();
  TORCH_MLU_CHECK(
      (ndim == 4 || ndim == 5),
      "cnnl_adaptive_avg_pool_internal(): Expected 4D or 5D tensor, but got ",
      input.sizes());
  TORCH_MLU_CHECK(
      input.scalar_type() == output.scalar_type(),
      "input and output must be the same dtype, output is ",
      output.scalar_type(),
      ", but input is ",
      input.scalar_type());

  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  if (ndim == 5)
    layout = CNNL_LAYOUT_NDHWC;
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();
  size_t ws_size = 0;
  TORCH_CNNL_CHECK(cnnlGetAdaptivePoolingForwardWorkspaceSize(
      handle,
      input_desc.get(),
      CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
      output_desc.get(),
      &ws_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(ws_size);

  // kernel calculate
  TORCH_CNNL_CHECK(cnnlAdaptivePoolingForward_v2(
      handle,
      input_desc.get(),
      input_ptr,
      CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
      ws_ptr.get(),
      ws_size,
      output_desc.get(),
      output_ptr,
      nullptr,
      nullptr));
}

void cnnl_adaptive_avg_pool_backward_internal(
    at::Tensor& gradInput,
    const at::Tensor& gradOutput_,
    const at::Tensor& input) {
  at::TensorArg grad_input_arg{gradInput, "gradInput", 1},
      grad_output_arg{gradOutput_, "gradOutput_", 2},
      input_arg{input, "input", 3};
  checkAllSameMLU(__func__, {grad_input_arg, grad_output_arg, input_arg});

  TORCH_MLU_CHECK(
      input.is_contiguous(at::MemoryFormat::ChannelsLast) ||
          input.is_contiguous(at::MemoryFormat::ChannelsLast3d),
      "cnnl_adaptive_avg_pool_internal(): only support channels "
      "last format.");
  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "cnnl_adaptive_avg_pool_backward_internal(): Expected 4D or 5D "
      "tensor, but got ",
      input.ndimension());
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
  if (input.dim() == 5)
    layout = CNNL_LAYOUT_NDHWC;
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(gradOutput_);
  auto output_impl = getMluTensorImpl(gradInput);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlAdaptivePoolingBackward(
      handle,
      input_desc.get(),
      input_ptr,
      nullptr,
      nullptr,
      CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
      output_desc.get(),
      output_ptr));
}

void cnnl_adaptive_max_pool2d_internal(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  // TODO(CNNLCORE-11573): remove this when cnnl support int32 index for half
  // dtype.
  if (input.scalar_type() == at::kHalf ||
      input.scalar_type() == at::kBFloat16) {
    indices = indices.to(at::kShort);
  } else {
    indices = indices.to(at::kInt);
  }
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto indices_impl = getMluTensorImpl(indices);
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_NHWC);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_NHWC);
  auto indices_desc = getTensorDesc(indices_impl, CNNL_LAYOUT_NHWC);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto indices_ptr = indices_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();
  size_t ws_size = 0;
  TORCH_CNNL_CHECK(cnnlGetAdaptivePoolingForwardWorkspaceSize(
      handle, input_desc.get(), CNNL_POOLING_MAX, output_desc.get(), &ws_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(ws_size);

  // kernel calculate
  TORCH_CNNL_CHECK(cnnlAdaptivePoolingForward_v2(
      handle,
      input_desc.get(),
      input_ptr,
      CNNL_POOLING_MAX,
      ws_ptr.get(),
      ws_size,
      output_desc.get(),
      output_ptr,
      indices_desc.get(),
      indices_ptr));
  // TODO(CNNLCORE-11573): remove this when cnnl support int32 index for half
  // dtype.
  indices = indices.to(at::kLong);
}

void cnnl_adaptive_max_pool2d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& indices) {
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(grad_output);
  auto indices_impl = getMluTensorImpl(indices);
  auto output_impl = getMluTensorImpl(grad_input);
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_NHWC);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_NHWC);
  auto indices_desc = getTensorDesc(indices_impl, CNNL_LAYOUT_NHWC);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto indices_ptr = indices_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlAdaptivePoolingBackward(
      handle,
      input_desc.get(),
      input_ptr,
      indices_desc.get(),
      indices_ptr,
      CNNL_POOLING_MAX,
      output_desc.get(),
      output_ptr));
}

} // namespace ops
} // namespace torch_mlu
