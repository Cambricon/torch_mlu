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
#include "ATen/native/Pool.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_max_unpool2d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  const int64_t kernel_h = kernel_size[0];
  const int64_t kernel_w = kernel_size.size() == 1 ? kernel_h : kernel_size[1];
  const int64_t stride_h = stride[0];
  const int64_t stride_w = stride.size() == 1 ? stride_h : stride[1];
  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding.size() == 1 ? pad_h : padding[1];
  auto pl = 0, pr = 0, pu = 0, pd = 0;
  pu = pd = pad_h;
  pl = pr = pad_w;

  auto input_impl = getMluTensorImpl(self);
  auto indices_impl = getMluTensorImpl(indices);
  auto output_impl = getMluTensorImpl(output);

  auto handle = getCurrentHandle();
  CnnlPoolingDescriptor pooling_desc;
  auto mode = CNNL_POOLING_MAX;
  pooling_desc.set(
      mode, kernel_h, kernel_w, stride_h, stride_w, pu, pd, pl, pr, false);
  CnnlTensorDescriptor input_desc, output_desc, indices_desc;
  input_desc.set(self, CNNL_LAYOUT_NHWC);
  output_desc.set(output, CNNL_LAYOUT_NHWC);
  indices_desc.set(indices, CNNL_LAYOUT_NHWC);

  void* indices_ptr = nullptr;
  if (indices.numel() > 0) {
    auto indices_impl = getMluTensorImpl(indices);
    indices_ptr = mlu_data_ptr(indices_impl);
  }
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  TORCH_CNNL_CHECK(cnnlUnpoolForward(
      handle,
      pooling_desc.desc(),
      input_desc.desc(),
      input_ptr,
      indices_desc.desc(),
      indices_ptr,
      output_desc.desc(),
      output_ptr));
  return output;
}

at::Tensor cnnl_max_unpool2d_backward_internal(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  const int64_t kernel_h = kernel_size[0];
  const int64_t kernel_w = kernel_size.size() == 1 ? kernel_h : kernel_size[1];
  const int64_t stride_h = stride[0];
  const int64_t stride_w = stride.size() == 1 ? stride_h : stride[1];
  const int64_t pad_h = padding[0];
  const int64_t pad_w = padding.size() == 1 ? pad_h : padding[1];
  auto pl = 0, pr = 0, pu = 0, pd = 0;
  pu = pd = pad_h;
  pl = pr = pad_w;

  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_input_impl = getMluTensorImpl(grad_input);

  auto handle = getCurrentHandle();
  CnnlPoolingDescriptor pooling_desc;
  auto mode = CNNL_POOLING_MAX;
  pooling_desc.set(
      mode, kernel_h, kernel_w, stride_h, stride_w, pu, pd, pl, pr, false);
  CnnlTensorDescriptor indices_desc, grad_output_desc, grad_input_desc;
  grad_output_desc.set(grad_output, CNNL_LAYOUT_NHWC);
  grad_input_desc.set(grad_input, CNNL_LAYOUT_NHWC);
  auto temp = at::empty(
      indices.sizes().vec(), indices.options(), at::MemoryFormat::ChannelsLast);
  indices_desc.set(temp, CNNL_LAYOUT_NHWC);

  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);
  void* indices_ptr = nullptr;

  auto indices_impl = getMluTensorImpl(indices);
  indices_ptr = mlu_data_ptr(indices_impl);

  TORCH_CNNL_CHECK(cnnlUnpoolBackward(
      handle,
      pooling_desc.desc(),
      grad_output_desc.desc(),
      grad_output_ptr,
      indices_desc.desc(),
      indices_ptr,
      grad_input_desc.desc(),
      grad_input_ptr));
  return grad_input;
}

} // namespace ops
} // namespace torch_mlu
