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
#include <ATen/TensorSubclassLikeUtils.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// Copy from pytorch/aten/src/ATen/native/LinearAlgebra.cpp.
// We modify should_fold for 3d @ 2d case, for mm is faster
// than bmm on MLU.
static bool should_fold(
    const Tensor& tensor1,
    const Tensor& tensor2,
    bool has_out) {
  // We check that we can fold the larger tensor into a matrix and dispatch to
  // mm or mv rather than to bmm. We want to make sure we can do so without
  // incurring in any extra copy
  const auto tensor1_larger = tensor1.dim() >= tensor2.dim();

  // We order the tensors. t1 will be the larger tensor
  // We can always transpose tensor2 as the dimensions are always >= 1
  // (precondition from matmul) and tensor1_larger iff tensor2.dim() >
  // tensor1.dim(9
  const auto t1 = tensor1_larger ? c10::MaybeOwned<Tensor>::borrowed(tensor1)
                                 : c10::MaybeOwned<Tensor>::owned(tensor2.mT());
  const int64_t dim_t1 = t1->dim();
  const auto dim_t2 = tensor1_larger ? tensor2.dim() : tensor1.dim();

  // Just fold for dim_t1 >= 3 and (dim_t2 == 1 || dim_t2 == 2)
  if (!(dim_t1 >= 3 && dim_t2 <= 2)) {
    return false;
  }

  // In this case we *do* incur in an extra copy to avoid creating an
  // unnecessary large tensor in the backward Suppose we don't fold here. Let
  // t1.shape = [b, m, n] t2.shape = [n, k] like in a transformer t2 will be
  // expanded to a tensor of shape [b, n, k] and then we do t1.bmm(t2_expanded)
  // The issue appears in the backward.
  // The output gradient g of this operation would have shape [b, m, k]
  // The backward wrt. t2 of bmm would be given by t1.mH @ g, which has shape
  // [b, n, k] Then, the backward of expand is simply `sum(0)`. As such, we are
  // instantiating a tensor of shape [b, n, k] unnecessarily, which may cause a
  // large memory footprint, and in the worst case, an OOM
  bool t2_requires_grad =
      tensor1_larger ? tensor2.requires_grad() : tensor1.requires_grad();
  if (t2_requires_grad && !has_out) {
    // We should be checking !at::GradMode::is_enabled(), but apparently
    // this regresses performance in some cases:
    // https://github.com/pytorch/pytorch/issues/118548#issuecomment-1916022394
    return true;
  }

  // Don't fold in this case, as we would have to call mm on the transposed
  // tensor, the result would be contiguous, and then we would need to transpose
  // it and call contiguous on it, thus having to copy the tensor
  if (tensor1.dim() == 2) {
    return false;
  }

  // DPF-3540: MLU just use mm instead of bmm for performance
  if (tensor1.device().is_privateuseone() ||
      tensor2.device().is_privateuseone()) {
    return true;
  }

  // Can always fold if the tensor is empty
  // This serves as a precondition for the code below
  if (t1->numel() == 0) {
    return true;
  }

  // t1->view(-1, t1->size(-1)) does not copy only when the first n-1 dimensions
  // are contiguous in the sense that t1_stride[i] =
  // t1_stride[i+1]*t1_shape[i+1]
  const auto t1_shape = t1->sizes();
  const auto t1_strides = t1->strides();
  for (auto i = int64_t{0}; i < dim_t1 - int64_t{2}; ++i) {
    if (t1_strides[i] != t1_strides[i + 1] * t1_shape[i + 1]) {
      return false;
    }
  }
  return true;
}

/*
Matrix product of two Tensors.
The behavior depends on the dimensionality of the Tensors as follows:
- If both Tensors are 1-dimensional, (1d) the dot product (scalar) is returned.
- If the arguments are 2D - 1D or 1D - 2D, the matrix-vector product is
returned.
- If both arguments are 2D, the matrix-matrix product is returned.
- If one of the arguments is ND with N >= 3 and the other is 1D or 2D, and some
  conditions on the strides apply (see should_fold) we fold the first N-1
dimensions of the ND argument to form a matrix, call mm or mv, reshape it back
to ND and return it
- Otherwise, we return bmm, after broadcasting and folding the batched
dimensions if there's more than one
*/
static Tensor _matmul_impl(
    Tensor& out,
    const Tensor& tensor1,
    const Tensor& tensor2) {
  at::NoNamesGuard guard;
  const auto dim_tensor1 = tensor1.dim();
  const auto dim_tensor2 = tensor2.dim();

  // This is checked up here to simplify the logic below
  // Note that the strings are just evaluated on failure, so almost always we
  // just evaluate the condition and move on
  TORCH_CHECK(
      dim_tensor1 != 0 && dim_tensor2 != 0,
      "both arguments to matmul need to be at least 1D, but they are ",
      dim_tensor1,
      "D and ",
      dim_tensor2,
      "D");

  const bool has_out = out.defined();

  if (has_out) {
    // Usually we would rely on the out= kernels we decompose into to check
    // this, but for matmul there is logic at the composite level that relies on
    // this invariant.
    TORCH_CHECK(
        !(tensor1.requires_grad() || tensor2.requires_grad() ||
          out.requires_grad()) ||
            !at::GradMode::is_enabled(),
        "matmul(): functions with out=... arguments don't support automatic differentiation, "
        "but one of the arguments requires grad.");
  }

  if (dim_tensor1 == 1 && dim_tensor2 == 1) {
    return has_out ? at::dot_out(out, tensor1, tensor2) : tensor1.dot(tensor2);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 1) {
    return has_out ? at::mv_out(out, tensor1, tensor2) : tensor1.mv(tensor2);
  } else if (dim_tensor1 == 1 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1.unsqueeze(0), tensor2).squeeze_(0)
                   : tensor1.unsqueeze(0).mm(tensor2).squeeze_(0);
  } else if (dim_tensor1 == 2 && dim_tensor2 == 2) {
    return has_out ? at::mm_out(out, tensor1, tensor2) : tensor1.mm(tensor2);
  } else if (should_fold(tensor1, tensor2, has_out)) {
    // dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2) ||
    // dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
    // and at least one of the following two conditions hold
    // - the small tensor requires grad (see should_fold for the why)
    // - we can fold the larger tensor t1 into a matrix as t1.view(-1,
    // t1.size(-1)) without copying

    // optimization: use mm instead of bmm by folding the batch of the larger
    // tensor into its leading matrix dimension
    const auto transpose = dim_tensor2 > dim_tensor1;
    const auto t1 = transpose ? c10::MaybeOwned<Tensor>::owned(tensor2.mT())
                              : c10::MaybeOwned<Tensor>::borrowed(tensor1);
    const auto t2 = !transpose ? c10::MaybeOwned<Tensor>::borrowed(tensor2)
        : dim_tensor1 == 2     ? c10::MaybeOwned<Tensor>::owned(tensor1.t())
                               : c10::MaybeOwned<Tensor>::borrowed(tensor1);
    // Invariant: t1->dim() >= 3 && (t2->dim() == 1 || t2->dim() == 2)
    //            and *t1 and *t2 are matmul-compatible

    // Why not t1->view(-1, sizes_1.back())?
    // If the last dim is 0, then view(-1, 0) won't work because the -1 becomes
    // ambiguous. This can happen in e.g. [3, 5, 0] @ [0, 0].
    const auto sizes_1 = t1->sizes();
    auto output_shape = c10::DimVector(sizes_1.begin(), sizes_1.end() - 1);
    const auto folded_dim1 = c10::multiply_integers(output_shape);

    // Readjust output_shape if we are multiplying by a matrix
    const auto t2_is_matrix = t2->dim() == 2;
    if (t2_is_matrix) {
      output_shape.push_back(t2->sizes()[1]);
    }
    // This will almost always be a view.
    // It may not be a view if t2->requires_grad(). See should_fold for an
    // explanation
    const auto t1_folded = t1->reshape({folded_dim1, sizes_1.back()});
    if (!has_out) {
      if (t2_is_matrix) {
        const auto output = at::_unsafe_view(t1_folded.mm(*t2), output_shape);
        // This copies if we perform a 2D @ 3D and the first tensor
        // requires_grad See should_fold for why. If mm_out were differentiable,
        // we could use it here, and pass a result with the correct strides to
        // avoid this unnecessary copy.
        return transpose ? output.mT().contiguous() : output;
      } else {
        return at::_unsafe_view(t1_folded.mv(*t2), output_shape);
      }
    } else {
      // See the !has_out branch for an explanation
      TORCH_INTERNAL_ASSERT(!(transpose && t2_is_matrix));

      // Resize output into the correct shape
      at::native::resize_output(out, output_shape);

      // We then reshape the output to the expected shape and call mm/mv
      // and transpose back if necessary
      auto reshaped_out = t2_is_matrix
          ? out.reshape({folded_dim1, t2->sizes().back()})
          : out.reshape({folded_dim1});
      if (t2_is_matrix) {
        at::mm_out(reshaped_out, t1_folded, *t2);
      } else {
        at::mv_out(reshaped_out, t1_folded, *t2);
      }
      if (!reshaped_out.is_alias_of(out)) {
        out.copy_(reshaped_out);
      }
      return out;
    }
  } else {
    // dim_tensor1 >= 3 || dim_tensor2 >= 3
    // We track m1 vs m2 separately even though they must match for nicer error
    // messages
    const int64_t n = dim_tensor1 > 1 ? tensor1.sizes().cend()[-2] : 1LL;
    const int64_t m1 = tensor1.sizes().back();
    auto batch_tensor1 =
        tensor1.sizes().slice(0, std::max<int64_t>(dim_tensor1 - 2, 0LL));
    const int64_t m2 =
        dim_tensor2 > 1 ? tensor2.sizes().cend()[-2] : tensor2.sizes().front();
    const int64_t p = dim_tensor2 > 1 ? tensor2.sizes().back() : 1LL;
    const IntArrayRef batch_tensor2(
        tensor2.sizes().data(), std::max<int64_t>(dim_tensor2 - 2, 0LL));

    // Same optimization for the gradients as that in should_fold
    // If we're going to broadcast we force it to go through the should_fold
    // branch
    if (dim_tensor1 == 3 && dim_tensor2 == 3 &&
        batch_tensor1[0] != batch_tensor2[0]) {
      if (batch_tensor1[0] == 1 &&
          (tensor1.requires_grad() || at::isTensorSubclassLike(tensor1))) {
        return _matmul_impl(out, tensor1.squeeze(0), tensor2);
      }
      if (batch_tensor2[0] == 1 &&
          (tensor2.requires_grad() || at::isTensorSubclassLike(tensor2))) {
        return _matmul_impl(out, tensor1, tensor2.squeeze(0));
      }
    }

    auto output_shape = at::infer_size_dimvector(batch_tensor1, batch_tensor2);
    const int64_t expand_batch_product = c10::multiply_integers(output_shape);

    // flatten expanded batches
    const auto tensor1_expand_size = [&output_shape, n, m1] {
      c10::DimVector ret(output_shape);
      ret.append({n, m1});
      return ret;
    }();
    const auto tensor1_expanded = tensor1.expand(tensor1_expand_size)
                                      .reshape({expand_batch_product, n, m1});
    // We need to treat the dim_tensor2 == 1 case separately as broadcasting
    // would not convert a vector of shape (n,) into a batch of matrices of
    // shape (*, n, 1)
    auto vector_rhs = dim_tensor2 == 1;
    const auto tensor2_expand_size = [&output_shape, m2, p, vector_rhs] {
      c10::DimVector ret(output_shape);
      if (vector_rhs) {
        ret.push_back(m2);
      } else {
        ret.append({m2, p});
      }
      return ret;
    }();
    auto tensor2_expanded = tensor2.expand(tensor2_expand_size);
    if (vector_rhs) {
      tensor2_expanded =
          tensor2_expanded.reshape({expand_batch_product, m2}).unsqueeze(2);
    } else {
      tensor2_expanded =
          tensor2_expanded.reshape({expand_batch_product, m2, p});
    }

    if (dim_tensor1 > 1) {
      output_shape.push_back(n);
    }
    if (dim_tensor2 > 1) {
      output_shape.push_back(p);
    }

    if (!has_out) {
      if (vector_rhs) {
        return at::_unsafe_view(
            tensor1_expanded.bmm(tensor2_expanded).squeeze(-1), output_shape);
      } else {
        return at::_unsafe_view(
            tensor1_expanded.bmm(tensor2_expanded), output_shape);
      }
    } else {
      at::native::resize_output(out, output_shape);
      auto reshaped_out = out.reshape({expand_batch_product, n, p});
      at::bmm_out(reshaped_out, tensor1_expanded, tensor2_expanded);
      if (vector_rhs) {
        reshaped_out = reshaped_out.squeeze(-1);
      }
      if (!reshaped_out.is_alias_of(out)) {
        out.copy_(reshaped_out.view_as(out));
      }
      return out;
    }
  }
}

Tensor cnnl_matmul(const Tensor& tensor1, const Tensor& tensor2) {
  auto maybe_outnames =
      at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  at::Tensor result, unused;
  result = _matmul_impl(unused, tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor cnnl_matmul_autograd(const Tensor& tensor1, const Tensor& tensor2) {
  return cnnl_matmul(tensor1, tensor2);
}

Tensor& cnnl_matmul_out(
    const Tensor& tensor1,
    const Tensor& tensor2,
    Tensor& result) {
  auto maybe_outnames =
      at::namedinference::compute_matmul_outnames(tensor1, tensor2);
  _matmul_impl(result, tensor1, tensor2);
  at::namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor& cnnl_matmul_out_out_autograd(
    const Tensor& tensor1,
    const Tensor& tensor2,
    Tensor& result) {
  return cnnl_matmul_out(tensor1, tensor2, result);
}

} //  namespace ops
} //  namespace torch_mlu
