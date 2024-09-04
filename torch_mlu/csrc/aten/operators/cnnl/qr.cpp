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
#include "ATen/native/LinearAlgebraUtils.h"

namespace torch_mlu {
namespace ops {

std::set<at::ScalarType> qr_support_dtype{
    at::ScalarType::Half,
    at::ScalarType::Float,
    at::ScalarType::BFloat16,
    at::ScalarType::Double};

typedef std::vector<int64_t> sizesType;

inline std::tuple<sizesType, sizesType> create_qr_output_sizes(
    const at::Tensor& input,
    c10::string_view mode) {
  bool some = true;
  if (mode == "complete")
    some = false;
  auto sizes = input.sizes().vec();
  int64_t m = input.size(-2), n = input.size(-1);

  sizesType Q_sizes;
  sizes[input.dim() - 1] = (some) ? std::min(m, n) : m;
  Q_sizes = sizes;
  sizesType R_sizes;
  sizes[input.dim() - 2] = (some) ? std::min(m, n) : m;
  sizes[input.dim() - 1] = n;
  R_sizes = sizes;
  return std::tuple<sizesType, sizesType>(Q_sizes, R_sizes);
}

inline std::tuple<at::Tensor&, at::Tensor&> cnnl__linalg_qr_helper(
    const at::Tensor& input,
    c10::string_view mode,
    at::Tensor& Q,
    at::Tensor& R) {
  at::native::checkIsMatrix(input, "linalg.qr");
  TORCH_CHECK(
      mode == "reduced" || mode == "r" || mode == "complete",
      "qr received unrecognized mode '",
      mode,
      "' but expected one of 'reduced' (default), 'r', or 'complete'");
  TORCH_CHECK(
      qr_support_dtype.find(input.scalar_type()) != qr_support_dtype.end(),
      "cnnl qr op not implemented for '",
      input.scalar_type(),
      "'");
  TORCH_CHECK(
      input.scalar_type() == Q.scalar_type(),
      "Expected out tensor to have dtype ",
      input.scalar_type(),
      ", but got ",
      Q.scalar_type(),
      " instead");
  TORCH_CHECK(
      input.scalar_type() == R.scalar_type(),
      "Expected out tensor to have dtype ",
      input.scalar_type(),
      ", but got ",
      R.scalar_type(),
      " instead");
  // get output Tensors
  auto self_contiguous = cnnl_contiguous(input);
  auto output_sizes = create_qr_output_sizes(self_contiguous, mode);
  auto Q_sizes = std::get<0>(output_sizes);
  auto R_sizes = std::get<1>(output_sizes);

  at::native::resize_output(R, R_sizes);
  bool some = true;
  if (mode == "complete")
    some = false;
  if (self_contiguous.numel() == 0) {
    at::Tensor Q_tmp;
    at::Tensor R_tmp = at::empty(R.sizes().vec(), self_contiguous.options());
    if (mode == "r") {
      Q_tmp = at::empty({0}, Q.options());
      at::native::resize_output(Q, Q_tmp.sizes());
    } else {
      at::native::resize_output(Q, Q_sizes);
      Q_tmp = at::eye(Q.size(-2), Q.size(-1), self_contiguous.options());
    }
    Q.copy_(Q_tmp);
    R.copy_(R_tmp);
  } else {
    at::native::resize_output(Q, Q_sizes);
    cnnl_qr_internal(Q, R, self_contiguous, some);
    if (mode == "r") {
      at::Tensor Q_tmp = at::empty({0}, Q.options());
      at::native::resize_output(Q, Q_tmp.sizes());
      Q.copy_(Q_tmp);
    }
  }
  return std::tuple<at::Tensor&, at::Tensor&>(Q, R);
}

std::tuple<at::Tensor, at::Tensor> cnnl_linalg_qr(
    const at::Tensor& A,
    c10::string_view mode) {
  auto Q = at::empty({0}, A.options());
  auto R = at::empty({0}, A.options());
  return cnnl_linalg_qr_out(A, mode, Q, R);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_linalg_qr_out(
    const at::Tensor& A,
    c10::string_view mode,
    at::Tensor& Q,
    at::Tensor& R) {
  cnnl__linalg_qr_helper(A, mode, Q, R);
  return std::tuple<at::Tensor&, at::Tensor&>(Q, R);
}

} // namespace ops
} // namespace torch_mlu
