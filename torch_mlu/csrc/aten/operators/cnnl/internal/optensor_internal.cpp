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

#include "ATen/OpMathType.h"
#include "aten/utils/dispatch.h"
#include "aten/utils/binaryops_util.h"
#include "aten/utils/types.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// CNNL OpTensor:    c = op(alpha1[0] * a, alpha2[0] * b) + beta[0] * c
// OpTensor alpha1 and alpha2 type is float when input and other type is not
// int, otherwise is int. self, other is not support scalar tensor.
at::Tensor cnnl_optensor_out_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha_scalar1,
    at::Scalar alpha_scalar2,
    at::Scalar beta_scalar,
    cnnlOpTensorDesc_t op_type) {
  if (self.numel() == 0 || other.numel() == 0) {
    return output;
  }
  auto handle = getCurrentHandle();

  auto self_impl = getMluTensorImpl(self);
  auto other_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(output);

  // get mlu ptr
  auto self_ptr = self_impl->mlu_data_ptr();
  auto other_ptr = other_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // get cnnl desc
  auto cnnl_suggest_layout = suggestCnnlLayout(output_impl);
  auto self_desc = getTensorDesc(self_impl, cnnl_suggest_layout);
  auto other_desc = getTensorDesc(other_impl, cnnl_suggest_layout);
  auto output_desc = getTensorDesc(output_impl, cnnl_suggest_layout);
  /* C = op(alpha1 * A, alpha2 * B) + beta * C */
  AT_DISPATCH_MLU_FLOAT_HALF_INT_COMPLEX_AND_BFLOAT16(
      output.scalar_type(), "optensor_internal", [&] {
        // workspace
        size_t workspace_size = 0;
        CnnlOpTensorDescriptor descOpTensor;
        auto output_type = getCnnlDataType(output.scalar_type());
        auto compute_type = (output_type == CNNL_DTYPE_BFLOAT16 ||
                             output_type == CNNL_DTYPE_HALF)
            ? CNNL_DTYPE_FLOAT
            : output_type;
        descOpTensor.set(op_type, compute_type, CNNL_NOT_PROPAGATE_NAN);
        using opmath_t = MLUAccumulateType_t<scalar_t>;
        auto alpha1_value = (alpha_scalar1).to<opmath_t>();
        auto alpha2_value = (alpha_scalar2).to<opmath_t>();
        auto beta_value = (beta_scalar).to<opmath_t>();

        TORCH_CNNL_CHECK(cnnlGetOpTensorWorkspaceSize_v2(
            handle,
            descOpTensor.desc(),
            &alpha1_value,
            self_desc.get(),
            self_ptr,
            &alpha2_value,
            other_desc.get(),
            other_ptr,
            &beta_value,
            output_desc.get(),
            output_ptr,
            &workspace_size));
        auto temp_ptr =
            torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

        TORCH_CNNL_CHECK(cnnlOpTensor(
            handle,
            descOpTensor.desc(),
            &alpha1_value,
            self_desc.get(),
            self_ptr,
            &alpha2_value,
            other_desc.get(),
            other_ptr,
            temp_ptr.get(),
            workspace_size,
            &beta_value,
            output_desc.get(),
            output_ptr));
      });
  return output;
}

// Support scalar tensor in opTensor API.
at::Tensor cnnl_optensor_out_with_scalar_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha_scalar1,
    at::Scalar alpha_scalar2,
    at::Scalar beta_scalar,
    cnnlOpTensorDesc_t op_type) {
  at::Tensor input_tensor =
      scalar_to_tensor_with_dtype(self, self.scalar_type());
  at::Tensor other_tensor =
      scalar_to_tensor_with_dtype(other, other.scalar_type());
  return cnnl_optensor_out_internal(
      output,
      input_tensor,
      other_tensor,
      alpha_scalar1,
      alpha_scalar2,
      beta_scalar,
      op_type);
}

} // namespace ops
} // namespace torch_mlu
