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

at::Tensor cnnl__cdist_forward(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const double p,
    std::optional<int64_t> compute_mode) {
  TORCH_CHECK(
      x1.dim() >= 2,
      "cdist only supports at least 2D tensors, X1 got: ",
      x1.dim(),
      "D");
  TORCH_CHECK(
      x2.dim() >= 2,
      "cdist only supports at least 2D tensors, X2 got: ",
      x2.dim(),
      "D");
  TORCH_CHECK(
      x1.size(-1) == x2.size(-1),
      "X1 and X2 must have the same number of columns. X1: ",
      x1.size(-1),
      " X2: ",
      x2.size(-1));

  TORCH_CHECK(
      at::isFloatingType(x1.scalar_type()),
      "cdist only supports floating-point dtypes, X1 got: ",
      x1.scalar_type());
  TORCH_CHECK(
      x1.scalar_type() != at::ScalarType::Half,
      "cnnl_cdist does not support half input for now, X1 got: ",
      x1.scalar_type());
  auto device1 = x1.device().type();
  TORCH_CHECK(
      device1 == c10::kPrivateUse1,
      "cnnl_cdist only supports MLU devices, X1 got: ",
      device1);
  TORCH_CHECK(
      at::isFloatingType(x2.scalar_type()),
      "cdist only supports floating-point dtypes, X2 got: ",
      x2.scalar_type());
  TORCH_CHECK(
      x2.scalar_type() != at::ScalarType::Half,
      "cnnl_cdist does not support half input for now, X2 got: ",
      x2.scalar_type());
  auto device2 = x2.device().type();
  TORCH_CHECK(
      device2 == c10::kPrivateUse1,
      "cnnl_cdist only supports MLU devices, X2 got: ",
      device2);

  int64_t c1 = x1.size(-1);
  int64_t c2 = x2.size(-1);
  // 0 - default value.
  // If p = 2 and r1 > 25 or r2 > 25 (these values are based on performance
  // metrics), it will try to compute distance using matrix multiplication
  // approach 1 - force to use matrix multiplication for p = 2 2 - do not use
  // matrix multiplication for p = 2
  int64_t mode = compute_mode.value_or(0);
  TORCH_CHECK(
      mode >= 0 && mode <= 2, "possible modes: 0, 1, 2, but was: ", mode);

  int64_t r1 = x1.size(-2);
  int64_t r2 = x2.size(-2);

  auto dim1 = x1.dim();
  auto dim2 = x2.dim();

  // For batch calculation we expand all dimensions(except the last two) to one,
  // with size that equals to product of them.
  // The last two dimensions will stay the same
  at::IntArrayRef batch_tensor1(x1.sizes().data(), dim1 - 2);
  at::IntArrayRef batch_tensor2(x2.sizes().data(), dim2 - 2);
  std::vector<int64_t> expand_batch_portion =
      at::infer_size(batch_tensor1, batch_tensor2);
  std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

  const int64_t expand_batch_product =
      c10::multiply_integers(expand_batch_portion);
  std::vector<int64_t> tensor1_view{expand_batch_product, r1, c1};
  std::vector<int64_t> tensor2_view{expand_batch_product, r2, c2};

  at::Tensor tensor1_expanded =
      x1.expand(tensor1_expand_size).contiguous().view(tensor1_view);
  at::Tensor tensor2_expanded =
      x2.expand(tensor2_expand_size).contiguous().view(tensor2_view);

  std::vector<int64_t> output_shape(expand_batch_portion);
  output_shape.insert(output_shape.end(), {r1, r2});
  std::vector<int64_t> output_shape_cnnl{expand_batch_product, r1, r2};

  at::Tensor result;
  auto tensor1_expanded_contiguous = cnnl_contiguous(tensor1_expanded);
  auto tensor2_expanded_contiguous = cnnl_contiguous(tensor2_expanded);
  // cnnlcdist only supports p = 1.0 for now
  if (r1 == 0 || r2 == 0 || expand_batch_product == 0) {
    result = at::empty(output_shape, x1.options());
  } else if (c1 == 0) {
    result = at::zeros(output_shape, x1.options());
  } else {
    result = at::empty(output_shape_cnnl, x1.options());
    auto result_contiguous = cnnl_contiguous(result);
    cnnl_cdist_forward_internal(
        result_contiguous,
        tensor1_expanded_contiguous,
        tensor2_expanded_contiguous,
        p);
    if (is_copy_necessary(result, result_contiguous)) {
      result.copy_(result_contiguous);
    }
    result.resize_(output_shape);
  }
  return result;
}

at::Tensor cnnl__cdist_backward(
    const at::Tensor& _grad,
    const at::Tensor& _x1,
    const at::Tensor& _x2,
    const double p,
    const at::Tensor& _cdist) {
  TORCH_CHECK(p == 1.0, "cnnl__cdist_backward only supports p = 1.0 for now");
  // Broadcasting might generate non-contiguous Tensors, so handle it before
  // doing checks
  int64_t c1 = _x1.size(-1);
  int64_t c2 = _x2.size(-1);
  int64_t r1 = _x1.size(-2);
  int64_t r2 = _x2.size(-2);
  auto dim1 = _x1.dim();
  auto dim2 = _x2.dim();
  at::IntArrayRef batch_tensor1(_x1.sizes().data(), dim1 - 2);
  at::IntArrayRef batch_tensor2(_x2.sizes().data(), dim2 - 2);
  std::vector<int64_t> expand_batch_portion =
      at::infer_size(batch_tensor1, batch_tensor2);
  std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});
  // Compute the linearized batch size
  const int64_t batch_product = c10::multiply_integers(expand_batch_portion);
  // Gracefully handle empty Tensors
  if (r1 == 0 || r2 == 0 || c1 == 0 || batch_product == 0) {
    return at::zeros_like(_x1, _x1.options());
  }
  at::Tensor x1 = _x1;
  if (tensor1_expand_size != x1.sizes()) {
    x1 = x1.expand(tensor1_expand_size).contiguous();
  }
  at::Tensor x2 = _x2;
  if (tensor2_expand_size != x2.sizes()) {
    x2 = x2.expand(tensor2_expand_size).contiguous();
  }

  x1 = x1.contiguous();
  x2 = x2.contiguous();
  auto cdist = _cdist.contiguous();
  auto grad = _grad.contiguous();

  int64_t n = x1.size(-2);
  int64_t m = x1.size(-1);
  auto device1 = x1.device().type();

  TORCH_CHECK(
      device1 == c10::kPrivateUse1,
      "cnnl__cdist_backward only supports MLU devices, X1 got: ",
      device1);
  auto device2 = x2.device().type();
  TORCH_CHECK(
      device2 == c10::kPrivateUse1,
      "cnnl__cdist_backward only supports MLU devices, X2 got: ",
      device2);

  at::Tensor grad_x1 =
      at::empty_like(x1, x1.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT)
          .view({batch_product, n, m});

  std::vector<int64_t> tensor1_view{batch_product, r1, c1};
  std::vector<int64_t> tensor2_view{batch_product, r2, c2};
  std::vector<int64_t> cdist_view{batch_product, r1, r2};
  at::Tensor tensor1_expanded =
      x1.expand(tensor1_expand_size).contiguous().view(tensor1_view);
  at::Tensor tensor2_expanded =
      x2.expand(tensor2_expand_size).contiguous().view(tensor2_view);
  at::Tensor cdist_viewed = cdist.contiguous().view(cdist_view);
  at::Tensor grad_viewed = grad.contiguous().view(cdist_view);
  cnnl_cdist_backward_internal(
      grad_x1,
      tensor1_expanded,
      tensor2_expanded,
      cdist_viewed,
      p,
      grad_viewed);
  return grad_x1.view(x1.sizes());
}

} // namespace ops
} // namespace torch_mlu
