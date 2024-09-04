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
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>

namespace torch_mlu {
namespace ops {

#define MAX_HEIGHT 150
#define MAX_WIDTH 150
/* torch.svd, implemented in terms of torch.linalg.svd. There are two main
   differences:
    1. the 2nd parameter is bool some=True, which if effectively the opposite
       of full_matrices=True
    2. svd returns V, while linalg.svd returns Vh = V^H;
       If compute_uv is False, torch.svd() returns zero-filled tensors for U and
   Vh, whereas torch.linalg.svd() returns empty tensors.
*/

void svd_mlu_kernel(
    const Tensor& A,
    const bool full_matrices,
    const bool compute_uv,
    const std::optional<c10::string_view>& driver,
    const Tensor& U,
    const Tensor& S,
    const Tensor& Vh,
    const Tensor& info) {
  auto size = A.sizes();
  auto dim = A.dim();
  TORCH_CHECK(
      size[dim - 1] < MAX_HEIGHT && size[dim - 2] < MAX_WIDTH,
      "The height or width of cnnl svd's input tensor should be less than 150");
  // TODO(CNNLCORE-13740): cnnl not support compute_uv's value  is false
  TORCH_CHECK(
      compute_uv, "torch_mlu svd not support compute_uv's value is false.");
  TORCH_CHECK(
      !driver.has_value(), "torch_mlu svd not support driver has value");
  at::Tensor s_contiguous = cnnl_contiguous(S);
  at::Tensor vh_contiguous = cnnl_contiguous(Vh);
  at::Tensor u_contiguous = cnnl_contiguous(U);
  std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> outputs(
      u_contiguous, s_contiguous, vh_contiguous);
  auto self_contiguous = cnnl_contiguous(A);
  cnnl_svd_internal(outputs, self_contiguous, !full_matrices, compute_uv, info);
  if (is_copy_necessary(U, u_contiguous)) {
    U.copy_(u_contiguous);
  }
  if (is_copy_necessary(S, s_contiguous)) {
    S.copy_(s_contiguous);
  }
  if (is_copy_necessary(Vh, vh_contiguous)) {
    Vh.copy_(vh_contiguous);
  }
}

// Copy from pytorch/aten/src/ATen/native/BatchLinearAlgebra.cpp.
// Override meta func and impl func and remove cuSolver for MLU.
// Row major strides are faster for MLU.
TORCH_META_FUNC(_linalg_svd_out_mlu)
(const Tensor& A,
 bool full_matrices,
 bool compute_uv,
 std::optional<c10::string_view> driver) {
  at::native::checkIsMatrix(A, "linalg.svd");
  at::native::checkFloatingOrComplex(A, "linalg.svd");

  auto sizes = A.sizes().vec();
  const auto m = sizes.cend()[-2];
  const auto n = sizes.cend()[-1];
  const auto k = std::min(m, n);

  // Prepare sizes for U
  if (compute_uv) {
    sizes.back() = full_matrices ? m : k;
    // We prefer row major strides for MLU.
    auto U_strides = at::native::batched_matrix_contiguous_strides(
        sizes, /*f-contig*=*/false);
    set_output_strided(0, sizes, U_strides, A.options(), {});

    // Prepare sizes for Vh
    sizes.end()[-2] = full_matrices ? n : k;
    sizes.end()[-1] = n;

    auto Vh_strides = at::native::batched_matrix_contiguous_strides(
        sizes, /*f-contig*=*/false);
    set_output_strided(2, sizes, Vh_strides, A.options(), {});
  } else {
    set_output_raw_strided(0, {0}, {}, A.options(), {});
    set_output_raw_strided(2, {0}, {}, A.options(), {});
  }

  // Prepare sizes for S. S is always real, even when A is complex.
  sizes.pop_back();
  sizes.end()[-1] = k;
  set_output_contiguous(
      1, sizes, A.options().dtype(c10::toRealValueType(A.scalar_type())), {});
}

TORCH_IMPL_FUNC(_linalg_svd_out_mlu)
(const Tensor& A,
 const bool full_matrices,
 const bool compute_uv,
 std::optional<c10::string_view> driver,
 const Tensor& U,
 const Tensor& S,
 const Tensor& Vh) {
  // Half optimisation half precondition for some parts of the LAPACK / cuSOLVER
  // In particular, the call to lapackSvd to compute lwork fails otherwise
  if (A.numel() == 0) {
    // Needed in the case that we have e.g. A.shape == (3, 0) and
    // full_matrices=True We fill U or Vh with the identity matrix as it's a
    // valid SVD for the empty matrix
    if (compute_uv && full_matrices) {
      if (U.numel() != 0) {
        U.zero_();
        U.diagonal(0, -2, -1).fill_(1.);
      }
      if (Vh.numel() != 0) {
        Vh.zero_();
        Vh.diagonal(0, -2, -1).fill_(1.);
      }
    }
    return;
  }

  // A always needs to be copied as its contents will be destroyed during the
  // computation of the SVD Now, MAGMA needs the copy to be on CPU, while
  // cuSOLVER needs it to be on CUDA, so we'll defer the copy as a column major
  // matrix to the backends.
  const auto info = at::zeros(
      IntArrayRef(A.sizes().begin(), A.sizes().end() - 2),
      A.options().dtype(c10::kInt));

  svd_mlu_kernel(A, full_matrices, compute_uv, driver, U, S, Vh, info);

  // TODO This should be removed, and the code checking for convergence should
  // be lifted from svd_cusolver to this function. We should then make sure that
  // this function never errors out.
  at::_linalg_check_errors(info, "linalg.svd", /*is_matrix*/ A.dim() == 2);
}

} // namespace ops
} // namespace torch_mlu
