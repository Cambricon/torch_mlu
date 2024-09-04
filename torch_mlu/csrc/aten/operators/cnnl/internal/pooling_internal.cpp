/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
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
#include "ATen/native/Pool.h"
#include "c10/core/ScalarType.h"
#include "cnnl.h"
#include <cstdint>

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_pool2d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    bool ceil_mode,
    bool count_include_pad,
    int64_t pool_mode_row,
    int dilationH,
    int dilationW) {
  auto output_size = output.sizes().vec();

  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  // get cnnl descriptor
  input_desc.set(self, CNNL_LAYOUT_NHWC);
  output_desc.set(output, CNNL_LAYOUT_NHWC);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = count_include_pad
      ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
      : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  if (pool_mode_row > 0)
    mode = CNNL_POOLING_MAX;

  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(
      mode,
      kH,
      kW,
      dH,
      dW,
      /*pad up   */ padH,
      /*pad down */ padH,
      /*pad left */ padW,
      /*pad right*/ padW,
      ceil_mode,
      dilationH,
      dilationW);
  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetPoolingWorkspaceSize_v2(
      handle,
      pooling_desc.desc(),
      input_desc.desc(),
      output_desc.desc(),
      &space_size));
  auto temp_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlPoolingForward(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y              */ output_ptr,
      /* workspace      */ temp_ptr.get(),
      /* workspace_size */ space_size));
  return output;
}

at::Tensor cnnl_pool2d_backward_internal(
    const at::Tensor& gradInput,
    const at::Tensor& gradOutput,
    const at::Tensor& self,
    const at::Tensor& index,
    const int64_t kH,
    const int64_t kW,
    const int64_t dH,
    const int64_t dW,
    const int64_t padH,
    const int64_t padW,
    bool ceil_mode,
    bool count_include_pad,
    int dilationH,
    int dilationW) {
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor gradOutput_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor index_desc;
  auto input_impl = getMluTensorImpl(self);
  auto gradOutput_impl = getMluTensorImpl(gradOutput);
  auto output_impl = getMluTensorImpl(gradInput);

  // get cnnl descriptor
  input_desc.set(self, CNNL_LAYOUT_NHWC);
  gradOutput_desc.set(gradOutput, CNNL_LAYOUT_NHWC);
  output_desc.set(gradInput, CNNL_LAYOUT_NHWC);
  auto temp = at::empty(
      gradOutput.sizes().vec(),
      gradOutput.options(),
      at::MemoryFormat::ChannelsLast);
  index_desc.set(temp, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT64);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto gradOutput_ptr = gradOutput_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  void* index_ptr = nullptr;

  // pooling mode
  cnnlPoolingMode_t mode = count_include_pad
      ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
      : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  if (index.numel() > 0) {
    mode = CNNL_POOLING_MAX;
    auto index_impl = getMluTensorImpl(index);
    index_ptr = index_impl->mlu_data_ptr();
  }
  const void* alpha = nullptr;
  const void* beta = nullptr;
  // PoolingBackward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(
      mode,
      kH,
      kW,
      dH,
      dW,
      padH,
      padH,
      padW,
      padW,
      ceil_mode,
      dilationH,
      dilationW);
  TORCH_CNNL_CHECK(cnnlPoolingBackward(
      /* handle       */ handle,
      /* pooling_desc */ pooling_desc.desc(),
      /* alpha        */ alpha,
      /* y_desc       */ index_desc.desc(),
      /* y            */ index_ptr,
      /* diff_y_desc  */ gradOutput_desc.desc(),
      /* diff_y       */ gradOutput_ptr,
      /* x_desc       */ input_desc.desc(),
      /* x            */ input_ptr,
      /* beta         */ beta,
      /* diff_x_desc  */ output_desc.desc(),
      /* diff_x       */ output_ptr));
  return gradInput;
}

at::Tensor cnnl_pool3d_internal(
    const at::Tensor& output,
    const at::Tensor& self,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool ceil_mode,
    bool count_include_pad,
    int64_t pool_mode_row,
    int dilationT,
    int dilationH,
    int dilationW) {
  int arrKernel[3], arrStride[3], arrPadding[6];
  int dilation[3] = {dilationT, dilationH, dilationW};
  arrKernel[0] = kT;
  arrKernel[1] = kH;
  arrKernel[2] = kW;
  arrStride[0] = dT;
  arrStride[1] = dH;
  arrStride[2] = dW;
  arrPadding[1] = arrPadding[0] = padT;
  arrPadding[3] = arrPadding[2] = padH;
  arrPadding[5] = arrPadding[4] = padW;

  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  // get cnnl descriptor
  input_desc.set(self, CNNL_LAYOUT_NDHWC);
  output_desc.set(output, CNNL_LAYOUT_NDHWC);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = count_include_pad
      ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
      : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  if (pool_mode_row > 0)
    mode = CNNL_POOLING_MAX;

  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(
      mode, self.dim(), arrKernel, arrStride, arrPadding, dilation, ceil_mode);
  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetPoolingWorkspaceSize_v2(
      handle,
      pooling_desc.desc(),
      input_desc.desc(),
      output_desc.desc(),
      &space_size));
  auto temp_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
  void* alpha = nullptr;
  void* beta = nullptr;
  void* workspace_ptr = nullptr;
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlPoolingForward(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y              */ output_ptr,
      /* workspace      */ temp_ptr.get(),
      /* workspace_size */ space_size));
  return output;
}

at::Tensor cnnl_pool3d_backward_internal(
    const at::Tensor& gradInput,
    const at::Tensor& gradOutput,
    const at::Tensor& self,
    const at::Tensor& indices,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool ceil_mode,
    bool count_include_pad,
    int dilationT,
    int dilationH,
    int dilationW) {
  int arrKernel[3], arrStride[3], arrPadding[6];
  int dilation[3] = {dilationT, dilationH, dilationW};
  arrKernel[0] = kT;
  arrKernel[1] = kH;
  arrKernel[2] = kW;
  arrStride[0] = dT;
  arrStride[1] = dH;
  arrStride[2] = dW;
  arrPadding[1] = arrPadding[0] = padT;
  arrPadding[3] = arrPadding[2] = padH;
  arrPadding[5] = arrPadding[4] = padW;
  cnnlTensorLayout_t layout = CNNL_LAYOUT_NDHWC;

  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor index_desc;
  CnnlTensorDescriptor grad_desc;
  CnnlTensorDescriptor output_desc;
  auto input_impl = getMluTensorImpl(self);
  auto grad_impl = getMluTensorImpl(gradOutput);
  auto output_impl = getMluTensorImpl(gradInput);

  input_desc.set(self, layout);
  grad_desc.set(gradOutput, layout);
  output_desc.set(gradInput, layout);
  auto temp = at::empty(
      gradOutput.sizes().vec(),
      gradOutput.options(),
      at::MemoryFormat::ChannelsLast3d);
  index_desc.set(temp, layout, CNNL_DTYPE_INT64);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto grad_ptr = grad_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  void* index_ptr = nullptr;
  // Determine the pooling mode
  cnnlPoolingMode_t mode = count_include_pad
      ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
      : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  if (indices.numel() > 0) {
    mode = CNNL_POOLING_MAX;
    auto index_impl = getMluTensorImpl(indices);
    index_ptr = index_impl->mlu_data_ptr();
  }

  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(
      mode, self.dim(), arrKernel, arrStride, arrPadding, dilation, ceil_mode);

  const void* alpha = nullptr;
  const void* beta = nullptr;

  TORCH_CNNL_CHECK(cnnlPoolingBackward(
      /* handle       */ handle,
      /* pooling_desc */ pooling_desc.desc(),
      /* alpha        */ alpha,
      /* y_desc       */ index_desc.desc(),
      /* y            */ index_ptr,
      /* diff_y_desc  */ grad_desc.desc(),
      /* diff_y       */ grad_ptr,
      /* x_desc       */ input_desc.desc(),
      /* x            */ input_ptr,
      /* beta         */ beta,
      /* diff_x_desc  */ output_desc.desc(),
      /* diff_x       */ output_ptr));

  return gradInput;
}

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool2d_with_indices_internal(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    bool ceil_mode,
    int dilationH,
    int dilationW) {
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto index_impl = getMluTensorImpl(indices);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor index_desc;

  // get cnnl descriptor
  input_desc.set(self, CNNL_LAYOUT_NHWC);
  output_desc.set(output, CNNL_LAYOUT_NHWC);
  index_desc.set(indices, CNNL_LAYOUT_NHWC, CNNL_DTYPE_INT64);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto index_ptr = index_impl->mlu_data_ptr();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = CNNL_POOLING_MAX;

  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetPoolingWithIndexWorkspaceSize_v2(
      handle,
      input_desc.desc(),
      output_desc.desc(),
      index_desc.desc(),
      &space_size));
  auto temp_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(
      mode,
      kH,
      kW,
      dH,
      dW,
      padH,
      padH,
      padW,
      padW,
      ceil_mode,
      dilationH,
      dilationW);
  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlPoolingForwardWithIndex(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y              */ output_ptr,
      /* index_desc     */ index_desc.desc(),
      /* index          */ index_ptr,
      /* workspace      */ temp_ptr.get(),
      /* workspace_size */ space_size));

  return std::make_tuple(output, indices);
}

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool3d_with_indices_internal(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool ceil_mode,
    int dilationT,
    int dilationH,
    int dilationW) {
  int arrKernel[3], arrStride[3], arrPadding[6];
  int dilation[3] = {dilationT, dilationH, dilationW};
  arrKernel[0] = kT;
  arrKernel[1] = kH;
  arrKernel[2] = kW;
  arrStride[0] = dT;
  arrStride[1] = dH;
  arrStride[2] = dW;
  arrPadding[1] = arrPadding[0] = padT;
  arrPadding[3] = arrPadding[2] = padH;
  arrPadding[5] = arrPadding[4] = padW;

  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto index_impl = getMluTensorImpl(indices);

  cnnlTensorLayout_t layout = CNNL_LAYOUT_NDHWC;
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor index_desc;
  input_desc.set(self, layout);
  output_desc.set(output, layout);
  index_desc.set(indices, layout, CNNL_DTYPE_INT64);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto index_ptr = index_impl->mlu_data_ptr();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = CNNL_POOLING_MAX;
  auto handle = getCurrentHandle();

  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(
      mode, 5, arrKernel, arrStride, arrPadding, dilation, ceil_mode);

  const void* alpha = nullptr;
  const void* beta = nullptr;
  void* workspace_ptr = nullptr;
  size_t workspace_size = 0;

  TORCH_CNNL_CHECK(cnnlPoolingForwardWithIndex(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.desc(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.desc(),
      /* y              */ output_ptr,
      /* index_desc     */ index_desc.desc(),
      /* index          */ index_ptr,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size));
  return std::make_tuple(output, indices);
}

} // namespace ops
} // namespace torch_mlu
