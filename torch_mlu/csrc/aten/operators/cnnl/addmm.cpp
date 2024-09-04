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

std::tuple<at::Tensor, bool> getMMInput(const at::Tensor& self) {
  bool is_trans_self;
  if ((!self.is_contiguous()) && (self.is_non_overlapping_and_dense()) &&
      (self.t().is_contiguous())) {
    is_trans_self = true;
    return std::make_tuple(self.t(), is_trans_self);
  } else {
    is_trans_self = false;
    return std::make_tuple(
        cnnl_contiguous(self, c10::MemoryFormat::Contiguous), is_trans_self);
  }
}

enum class Activation {
  None,
  RELU,
  GELU,
};

at::Tensor& addmm_out_mlu_impl(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    Activation activation = Activation::None) {
  TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "tensors must be 2-D");

  at::TensorArg args[]{
      {result, "out", 0},
      {self, "self", 1},
      {mat1, "mat1", 2},
      {mat2, "mat2", 3}};
  checkAllSameMLU(__func__, args);

  at::IntArrayRef mat1_sizes = mat1.sizes();
  at::IntArrayRef mat2_sizes = mat2.sizes();
  at::IntArrayRef self__sizes;
  c10::MaybeOwned<at::Tensor> self_;
  bool useLtInterface = false;
  at::ScalarType scalar_type = self.scalar_type();
  if (&result != &self) {
    useLtInterface = beta.toComplexDouble() == 1.0 && self.dim() == 1 &&
        result.dim() == 2 && self.sizes()[0] == mat2_sizes[1] &&
        self.is_contiguous() &&
        (scalar_type == at::ScalarType::Float ||
         scalar_type == at::ScalarType::Half ||
         scalar_type == at::ScalarType::BFloat16);
    if (!useLtInterface) {
      self_ = at::expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    }
    self__sizes = self_->sizes();
  } else {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self__sizes = self_->sizes();
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(
        self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(
        self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  // NB: Here we do not need to copy self to result like CUDA
  // because CNNL kernel can handle the case.

  at::IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  // for some cases, GPU and CPU have different results, and MLU
  // results are same with GPU. For example, [b] + [a, 0] x [0, b],
  // CPU result's shape is [a, b], GPU and MLU result's shape is [b]
  if (mat1.numel() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    return at::mul_out(
        result,
        self,
        at::native::scalar_tensor(
            beta,
            self.scalar_type(),
            c10::nullopt /* layout */,
            at::kCPU,
            c10::nullopt /* pin_memory */));
  }

  TORCH_CHECK(
      scalar_type == mat1.scalar_type(),
      "expected scalar type ",
      scalar_type,
      " but found ",
      mat1.scalar_type());
  TORCH_CHECK(
      scalar_type == mat2.scalar_type(),
      "expected scalar type ",
      scalar_type,
      " but found ",
      mat2.scalar_type());
  TORCH_CHECK(
      scalar_type == result.scalar_type(),
      "expected scalar type ",
      scalar_type,
      " but found ",
      result.scalar_type());

  at::Tensor mat1_contiguous;
  at::Tensor mat2_contiguous;
  bool is_trans_mat1;
  bool is_trans_mat2;

  // acquire transposed matrix in some cases for better performance
  std::tie(mat1_contiguous, is_trans_mat1) = getMMInput(mat1);
  std::tie(mat2_contiguous, is_trans_mat2) = getMMInput(mat2);
  auto result_contiguous = cnnl_contiguous(result);
  at::Tensor self_contiguous;

  if (useLtInterface) {
    // bias should be unsqueeze to 2 dims for broadcast
    self_contiguous = cnnl_contiguous(at::unsqueeze(self, 0));
    cnnl_addmm_bias_out_internal(
        result_contiguous,
        self_contiguous,
        mat1_contiguous,
        mat2_contiguous,
        false,
        is_trans_mat1,
        is_trans_mat2,
        beta,
        alpha,
        !at::NoTF32Guard::should_disable_tf32() &&
            at::globalContext().allowTF32CnMatMul());
  } else {
    self_contiguous = cnnl_contiguous(*self_);
    cnnl_addmm_out_internal(
        result_contiguous,
        self_contiguous,
        mat1_contiguous,
        mat2_contiguous,
        false,
        is_trans_mat1,
        is_trans_mat2,
        beta,
        alpha,
        !at::NoTF32Guard::should_disable_tf32() &&
            at::globalContext().allowTF32CnMatMul());
  }
  switch (activation) {
    case Activation::RELU:
      at::relu_(const_cast<at::Tensor&>(result_contiguous));
      break;
    case Activation::GELU:
      at::gelu_(const_cast<at::Tensor&>(result_contiguous));
      break;
    default:
      break;
  }

  if (!result.is_same(result_contiguous)) {
    result.copy_(result_contiguous);
  }
  return result;
}

TORCH_IMPL_FUNC(mm_out_mlu)
(const at::Tensor& self, const at::Tensor& mat2, const at::Tensor& result) {
  addmm_out_mlu_impl(const_cast<at::Tensor&>(result), result, self, mat2, 0, 1);
}

TORCH_IMPL_FUNC(addmm_out_mlu)
(const at::Tensor& self,
 const at::Tensor& mat1,
 const at::Tensor& mat2,
 const at::Scalar& beta,
 const at::Scalar& alpha,
 const at::Tensor& result) {
  addmm_out_mlu_impl(
      const_cast<at::Tensor&>(result), self, mat1, mat2, beta, alpha);
}

TORCH_IMPL_FUNC(_addmm_activation_out_mlu)
(const at::Tensor& self,
 const at::Tensor& mat1,
 const at::Tensor& mat2,
 const at::Scalar& beta,
 const at::Scalar& alpha,
 bool use_gelu,
 const at::Tensor& result) {
  addmm_out_mlu_impl(
      const_cast<at::Tensor&>(result),
      self,
      mat1,
      mat2,
      beta,
      alpha,
      use_gelu ? Activation::GELU : Activation::RELU);
}

} // namespace ops
} // namespace torch_mlu
