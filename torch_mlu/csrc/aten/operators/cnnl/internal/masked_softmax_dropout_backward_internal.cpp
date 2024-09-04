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

#include "cnnl_extra.h"

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_masked_softmax_dropout_backward_internal(
    at::Tensor& diff_x,
    const at::Tensor& softmax_out,
    const at::Tensor& diff_y,
    const at::Tensor& dropout_mask,
    const int axis,
    const float p) {
  auto diff_x_impl = getMluTensorImpl(diff_x);
  auto desc_diff_x = getTensorDesc(diff_x_impl, CNNL_LAYOUT_ARRAY);
  auto diff_x_ptr = mlu_data_ptr(diff_x_impl);

  auto diff_y_impl = getMluTensorImpl(diff_y);
  auto desc_diff_y = getTensorDesc(diff_y_impl, CNNL_LAYOUT_ARRAY);
  auto diff_y_ptr = mlu_data_ptr(diff_y_impl);

  auto softmax_out_impl = getMluTensorImpl(softmax_out);
  auto desc_softmax_out = getTensorDesc(softmax_out_impl, CNNL_LAYOUT_ARRAY);
  auto softmax_out_ptr = mlu_data_ptr(softmax_out_impl);

  auto dropout_mask_impl = getMluTensorImpl(dropout_mask);
  auto desc_dropout_mask = getTensorDesc(dropout_mask_impl, CNNL_LAYOUT_ARRAY);
  auto dropout_mask_ptr = mlu_data_ptr(dropout_mask_impl);

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlMaskedScaleSoftmaxBackward(
      handle,
      axis,
      1.0,
      desc_softmax_out.get(),
      softmax_out_ptr,
      desc_diff_y.get(),
      diff_y_ptr,
      desc_dropout_mask.get(),
      static_cast<const uint8_t*>(dropout_mask_ptr),
      p,
      desc_diff_x.get(),
      diff_x_ptr));

  return diff_x;
}

} // namespace ops
} // namespace torch_mlu
