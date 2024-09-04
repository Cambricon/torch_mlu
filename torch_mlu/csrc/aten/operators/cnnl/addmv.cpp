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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

TORCH_IMPL_FUNC(addmv_out_mlu)
(const at::Tensor& self,
 const at::Tensor& mat,
 const at::Tensor& vec,
 const at::Scalar& beta_,
 const at::Scalar& alpha_,
 const at::Tensor& result) {
  // self is 1-dim vector or 0-dim scalar
  c10::MaybeOwned<at::Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta_.toComplexDouble();
  if (mat.numel() == 0) {
    // shortcut for an empty matrix
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (betaval == 0.0) {
      result.zero_();
    } else {
      at::mul_out(
          const_cast<at::Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta_,
              self.scalar_type(),
              c10::nullopt /* layout */,
              at::kCPU,
              c10::nullopt /* pin_memory */));
    }
  } else {
    // NB: Here we do not need to copy self to result like CUDA
    // because CNNL kernel can handle the case.
    if (result.numel() != 0) {
      at::ScalarType scalar_type = mat.scalar_type();
      TORCH_CHECK(
          scalar_type == vec.scalar_type(),
          "expected scalar type ",
          scalar_type,
          " but found ",
          vec.scalar_type());
      TORCH_CHECK(
          scalar_type == result.scalar_type(),
          "expected scalar type ",
          scalar_type,
          " but found ",
          result.scalar_type());

      // use addmm to realize the addmv, unsqueeze to 2-dim tensor
      auto self_contiguous = cnnl_contiguous((*self_).unsqueeze(1));
      auto mat_contiguous = cnnl_contiguous(mat);
      auto vec_contiguous = cnnl_contiguous(vec.unsqueeze(1));
      auto result_contiguous = cnnl_contiguous(result.unsqueeze(1));
      cnnl_addmm_out_internal(
          result_contiguous,
          self_contiguous,
          mat_contiguous,
          vec_contiguous,
          false,
          false,
          false,
          beta_,
          alpha_,
          torch_mlu::Global::instance().allowMLUCustomTF32());
      result_contiguous = result_contiguous.squeeze(1);
      if (is_copy_necessary(result, result_contiguous)) {
        result.copy_(result_contiguous);
      }
    }
  }
}

} // namespace ops
} // namespace torch_mlu
