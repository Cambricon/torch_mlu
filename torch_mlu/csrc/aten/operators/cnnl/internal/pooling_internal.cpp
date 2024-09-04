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
    int64_t pool_mode_row) {
  auto input_impl = getMluTensorImpl(self);
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_NHWC);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_NHWC);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

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
      ceil_mode);
  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetPoolingWorkspaceSize_v2(
      handle,
      pooling_desc.desc(),
      input_desc.get(),
      output_desc.get(),
      &space_size));
  auto temp_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlPoolingForward(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.get(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.get(),
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
    bool count_include_pad) {
  // get current handle
  auto handle = getCurrentHandle();

  auto input_impl = getMluTensorImpl(self);
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_NHWC);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto gradOutput_impl = getMluTensorImpl(gradOutput);
  auto gradOutput_desc = getTensorDesc(gradOutput_impl, CNNL_LAYOUT_NHWC);
  auto gradOutput_ptr = mlu_data_ptr(gradOutput_impl);

  auto output_impl = getMluTensorImpl(gradInput);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_NHWC);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto temp = at::empty(
      gradOutput.sizes(), gradOutput.options(), at::MemoryFormat::ChannelsLast);
  auto temp_impl = getMluTensorImpl(temp);
  // ??? >_< ???
  tensorDescPtr_t index_desc =
      getTensorDesc(temp_impl, CNNL_DTYPE_INT64, CNNL_LAYOUT_NHWC);

  // pooling mode
  cnnlPoolingMode_t mode = count_include_pad
      ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
      : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  void* index_ptr = nullptr;
  if (index.numel() > 0) {
    mode = CNNL_POOLING_MAX;
    auto index_impl = getMluTensorImpl(index);
    index_ptr = mlu_data_ptr(index_impl);
  }
  const void* alpha = nullptr;
  const void* beta = nullptr;
  // PoolingBackward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(mode, kH, kW, dH, dW, padH, padH, padW, padW, ceil_mode);
  TORCH_CNNL_CHECK(cnnlPoolingBackward(
      /* handle       */ handle,
      /* pooling_desc */ pooling_desc.desc(),
      /* alpha        */ alpha,
      /* y_desc       */ index_desc.get(),
      /* y            */ index_ptr,
      /* diff_y_desc  */ gradOutput_desc.get(),
      /* diff_y       */ gradOutput_ptr,
      /* x_desc       */ input_desc.get(),
      /* x            */ input_ptr,
      /* beta         */ beta,
      /* diff_x_desc  */ output_desc.get(),
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
    int64_t pool_mode_row) {
  int arrKernel[3], arrStride[3], arrPadding[6];
  int dilation[3] = {1, 1, 1};
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
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_NDHWC);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_NDHWC);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

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
      input_desc.get(),
      output_desc.get(),
      &space_size));
  auto temp_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
  void* alpha = nullptr;
  void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlPoolingForward(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.get(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.get(),
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
    bool count_include_pad) {
  int arrKernel[3], arrStride[3], arrPadding[6];
  int dilation[3] = {1, 1, 1};
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

  auto input_impl = getMluTensorImpl(self);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto grad_impl = getMluTensorImpl(gradOutput);
  auto grad_desc = getTensorDesc(grad_impl, layout);
  auto grad_ptr = mlu_data_ptr(grad_impl);

  auto output_impl = getMluTensorImpl(gradInput);
  auto output_desc = getTensorDesc(output_impl, layout);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto temp = at::empty(
      gradOutput.sizes().vec(),
      gradOutput.options(),
      at::MemoryFormat::ChannelsLast3d);
  auto temp_impl = getMluTensorImpl(temp);
  // ??? >_< ???
  tensorDescPtr_t index_desc =
      getTensorDesc(temp_impl, CNNL_DTYPE_INT64, CNNL_LAYOUT_NDHWC);

  void* index_ptr = nullptr;
  // Determine the pooling mode
  cnnlPoolingMode_t mode = count_include_pad
      ? CNNL_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
      : CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

  if (indices.numel() > 0) {
    mode = CNNL_POOLING_MAX;
    auto index_impl = getMluTensorImpl(indices);
    index_ptr = mlu_data_ptr(index_impl);
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
      /* y_desc       */ index_desc.get(),
      /* y            */ index_ptr,
      /* diff_y_desc  */ grad_desc.get(),
      /* diff_y       */ grad_ptr,
      /* x_desc       */ input_desc.get(),
      /* x            */ input_ptr,
      /* beta         */ beta,
      /* diff_x_desc  */ output_desc.get(),
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
    bool ceil_mode) {
  auto input_impl = getMluTensorImpl(self);
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_NHWC);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_NHWC);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto index_impl = getMluTensorImpl(indices);
  auto index_desc =
      getTensorDesc(index_impl, CNNL_DTYPE_INT64, CNNL_LAYOUT_NHWC);
  auto index_ptr = mlu_data_ptr(index_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // Determine the pooling mode
  cnnlPoolingMode_t mode = CNNL_POOLING_MAX;

  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetPoolingWithIndexWorkspaceSize_v2(
      handle,
      input_desc.get(),
      output_desc.get(),
      index_desc.get(),
      &space_size));
  auto temp_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  // pooling forward
  CnnlPoolingDescriptor pooling_desc;
  pooling_desc.set(mode, kH, kW, dH, dW, padH, padH, padW, padW, ceil_mode);
  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlPoolingForwardWithIndex(
      /* handle         */ handle,
      /* pooling_desc   */ pooling_desc.desc(),
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.get(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.get(),
      /* y              */ output_ptr,
      /* index_desc     */ index_desc.get(),
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
    bool ceil_mode) {
  int arrKernel[3], arrStride[3], arrPadding[6];
  int dilation[3] = {1, 1, 1};
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

  auto input_impl = getMluTensorImpl(self);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, layout);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto index_impl = getMluTensorImpl(indices);
  auto index_desc = getTensorDesc(index_impl, CNNL_DTYPE_INT64, layout);
  auto index_ptr = mlu_data_ptr(index_impl);

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
      /* x_desc         */ input_desc.get(),
      /* x              */ input_ptr,
      /* beta           */ beta,
      /* y_desc         */ output_desc.get(),
      /* y              */ output_ptr,
      /* index_desc     */ index_desc.get(),
      /* index          */ index_ptr,
      /* workspace      */ workspace_ptr,
      /* workspace_size */ workspace_size));
  return std::make_tuple(output, indices);
}

} // namespace ops
} // namespace torch_mlu
