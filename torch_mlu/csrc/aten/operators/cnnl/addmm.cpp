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
namespace {
// TODO:
// https://github.com/pytorch/pytorch/pull/59380#pullrequestreview-725310492
c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(
    const Tensor& tensor,
    bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cnnl(
    const Tensor& tensor,
    bool& transpose_tensor,
    bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
    transpose_tensor = !tensor.is_contiguous();
    return resolve_conj_if_indicated(
        tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[1] == 1) &&
      (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if (
      (tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = false;
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_cnnl(
    const Tensor& tensor,
    bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
    transpose_tensor = !tensor.is_contiguous();
    return resolve_conj_if_indicated(tensor, true);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[1] == 1) &&
      (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if (
      (tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = false;
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

struct cnCommonArgs {
  cnCommonArgs(const Tensor& mat1, const Tensor& mat2, Tensor& c) {
    bool transpose_result, transpose_mat1, transpose_mat2;
    result = prepare_matrix_for_cnnl(c, transpose_result);
    mata = prepare_matrix_for_cnnl(
        transpose_result ? mat2 : mat1, transpose_mat1, transpose_result);
    matb = prepare_matrix_for_cnnl(
        transpose_result ? mat1 : mat2, transpose_mat2, transpose_result);
    auto mat1_sizes = mat1.sizes();
    auto mat2_sizes = mat2.sizes();
    if (transpose_result) {
      transpose_mat1 = !transpose_mat1;
      transpose_mat2 = !transpose_mat2;
      mat1_sizes = mata->sizes();
      mat2_sizes = matb->sizes();
    }

    m = mat1_sizes[transpose_result ? 1 : 0];
    k = mat1_sizes[transpose_result ? 0 : 1];
    n = mat2_sizes[transpose_result ? 0 : 1];
    lda = mata->stride((transpose_mat1 == transpose_result) ? 0 : 1);
    ldb = matb->stride((transpose_mat2 == transpose_result) ? 0 : 1);
    result_ld = result->stride(transpose_result ? 1 : 0);
    transa = transpose_mat1 ? 1 : 0;
    transb = transpose_mat2 ? 1 : 0;
    trans_result = transpose_result;
  }
  bool transa, transb, trans_result;
  int64_t m, n, k;
  int64_t lda, ldb, result_ld;
  c10::MaybeOwned<Tensor> mata, matb, result;
};
} // namespace

at::Tensor getMMInput(const at::Tensor& self, const bool& trans) {
  if (trans) {
    return self.t();
  } else {
    return self;
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

  at::TensorArg targs[]{
      {result, "out", 0},
      {self, "self", 1},
      {mat1, "mat1", 2},
      {mat2, "mat2", 3}};
  checkAllSameMLU(__func__, targs);

  at::IntArrayRef mat1_sizes = mat1.sizes();
  at::IntArrayRef mat2_sizes = mat2.sizes();
  at::IntArrayRef self__sizes;
  c10::MaybeOwned<at::Tensor> self_;
  bool useExInterface = false;
  at::ScalarType scalar_type = self.scalar_type();
  if (&result != &self) {
    useExInterface = beta.toComplexDouble() == 1.0 && self.dim() == 1 &&
        result.dim() == 2 && self.sizes()[0] == mat2_sizes[1] &&
        self.is_contiguous() && result.is_contiguous() &&
        (scalar_type == at::ScalarType::Float ||
         scalar_type == at::ScalarType::Half ||
         scalar_type == at::ScalarType::BFloat16);
    if (!useExInterface) {
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

  if (&result != &self) {
    at::native::resize_output(result, {mat1_sizes[0], mat2_sizes[1]});
    if (beta.toComplexDouble() != 0.0 && !useExInterface) {
      cnnl_copy_internal(result, *self_);
    }
  }

  at::IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  cnCommonArgs args(mat1, mat2, result);

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
        self.expand(result.sizes()),
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

  at::Tensor mata_tensor = *args.mata;
  at::Tensor matb_tensor = *args.matb;
  at::Tensor result_tensor = *args.result;
  mata_tensor = getMMInput(mata_tensor, (args.transa != args.trans_result));
  matb_tensor = getMMInput(matb_tensor, (args.transb != args.trans_result));
  result_tensor = getMMInput(result_tensor, args.trans_result);
  at::Tensor self_tensor;

  if (useExInterface) {
    cnnlActivationMode_t mode = CNNL_ACTIVATION_IDENTITY;
    switch (activation) {
      case Activation::RELU:
        mode = CNNL_ACTIVATION_RELU;
        break;
      case Activation::GELU:
        mode = CNNL_ACTIVATION_GELU;
        break;
      default:
        break;
    }
    // bias should be unsqueeze to 2 dims for broadcast
    self_tensor =
        args.trans_result ? at::unsqueeze(self, 1) : at::unsqueeze(self, 0);
    cnnl_addmm_bias_out_internal(
        result_tensor,
        self_tensor,
        mata_tensor,
        matb_tensor,
        false,
        args.transa,
        args.transb,
        beta,
        alpha,
        mode,
        !at::NoTF32Guard::should_disable_tf32() &&
            torch_mlu::Global::instance().allowTF32CnMatMul());
  } else {
    cnnl_addmm_out_internal(
        result_tensor,
        result_tensor,
        mata_tensor,
        matb_tensor,
        false,
        args.transa,
        args.transb,
        beta,
        alpha,
        !at::NoTF32Guard::should_disable_tf32() &&
            torch_mlu::Global::instance().allowTF32CnMatMul());
    switch (activation) {
      case Activation::RELU:
        at::relu_(const_cast<at::Tensor&>(result_tensor));
        break;
      case Activation::GELU:
        at::gelu_(const_cast<at::Tensor&>(result_tensor), "tanh");
        break;
      default:
        break;
    }
  }

  if (!result.is_same(*args.result)) {
    result.copy_(*args.result);
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
