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

void cnnl_weight_norm_backward_internal(
    at::Tensor& grad_v,
    at::Tensor& grad_g,
    const at::Tensor& grad_w,
    const at::Tensor& saved_v,
    const at::Tensor& saved_g,
    const at::Tensor& saved_norms,
    const int64_t dim) {
  // [in] handle
  auto handle = getCurrentHandle();

  // [in] diff_w_desc & [in] diff_w
  CnnlTensorDescriptor grad_w_desc;
  grad_w_desc.set(grad_w);
  auto grad_w_impl = getMluTensorImpl(grad_w);
  auto grad_w_ptr = mlu_data_ptr(grad_w_impl);

  // [in] v_desc & [in] v
  CnnlTensorDescriptor saved_v_desc;
  saved_v_desc.set(saved_v);
  auto saved_v_impl = getMluTensorImpl(saved_v);
  auto saved_v_ptr = mlu_data_ptr(saved_v_impl);

  // [in] g_desc & [in] g
  CnnlTensorDescriptor saved_g_desc;
  saved_g_desc.set(saved_g);
  auto saved_g_impl = getMluTensorImpl(saved_g);
  auto saved_g_ptr = mlu_data_ptr(saved_g_impl);

  // [in] norm_recip_desc & [in] norm_recip
  CnnlTensorDescriptor saved_norms_desc;
  saved_norms_desc.set(saved_norms);
  auto saved_norms_impl = getMluTensorImpl(saved_norms);
  auto saved_norms_ptr = mlu_data_ptr(saved_norms_impl);

  // [in] axis
  const int axis = static_cast<int>(dim);

  // [in] diff_v_desc & [out] diff_v
  CnnlTensorDescriptor grad_v_desc;
  grad_v_desc.set(grad_v);
  auto grad_v_impl = getMluTensorImpl(grad_v);
  auto grad_v_ptr = mlu_data_ptr(grad_v_impl);

  // [in] diff_g_desc & [out] diff_g
  CnnlTensorDescriptor grad_g_desc;
  grad_g_desc.set(grad_g);
  auto grad_g_impl = getMluTensorImpl(grad_g);
  auto grad_g_ptr = mlu_data_ptr(grad_g_impl);

  // WeightNormBackward operation
  TORCH_CNNL_CHECK(cnnlWeightNormBackward(
      /*handle         */ handle,
      /*diff_w_desc    */ grad_w_desc.desc(),
      /*diff_w         */ grad_w_ptr,
      /*v_desc         */ saved_v_desc.desc(),
      /*v              */ saved_v_ptr,
      /*g_desc         */ saved_g_desc.desc(),
      /*g              */ saved_g_ptr,
      /*norm_recip_desc*/ saved_norms_desc.desc(),
      /*norm_recip     */ saved_norms_ptr,
      /*axis           */ axis,
      /*diff_v_desc    */ grad_v_desc.desc(),
      /*diff_v         */ grad_v_ptr,
      /*diff_g_desc    */ grad_g_desc.desc(),
      /*diff_g         */ grad_g_ptr));

  return;
}

} // namespace ops
} // namespace torch_mlu
