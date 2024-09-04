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

at::Tensor& cnnl_replication_pad1d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef padding) {
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_output_impl = getMluTensorImpl(grad_output);
  int pad[2];
  for (int i = 0; i < padding.size(); i++) {
    pad[i] = static_cast<int>(padding[i]);
  }
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_grad_input;
  CnnlTensorDescriptor desc_grad_output;
  // only support CNNL_LAYOUT_NCL
  desc_grad_input.set(grad_input, CNNL_LAYOUT_NLC);
  desc_grad_output.set(grad_output, CNNL_LAYOUT_NLC);
  // get onchip dataptr
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);
  TORCH_CNNL_CHECK(cnnlReplicationPadBackward(
      handle,
      desc_grad_output.desc(),
      grad_output_ptr,
      pad,
      desc_grad_input.desc(),
      grad_input_ptr));
  return grad_input;
}

at::Tensor& cnnl_replication_pad2d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef padding) {
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_output_impl = getMluTensorImpl(grad_output);
  int pad[4];
  for (int i = 0; i < padding.size(); i++) {
    pad[i] = static_cast<int>(padding[i]);
  }
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_grad_input;
  CnnlTensorDescriptor desc_grad_output;
  // only support CNNL_LAYOUT_NCHW
  desc_grad_input.set(grad_input, CNNL_LAYOUT_NHWC);
  desc_grad_output.set(grad_output, CNNL_LAYOUT_NHWC);
  // get onchip dataptr
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);
  TORCH_CNNL_CHECK(cnnlReplicationPadBackward(
      handle,
      desc_grad_output.desc(),
      grad_output_ptr,
      pad,
      desc_grad_input.desc(),
      grad_input_ptr));
  return grad_input;
}
} // namespace ops
} // namespace torch_mlu
