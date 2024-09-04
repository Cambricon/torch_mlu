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

#include <ATen/native/UnaryOps.h>
#include <ATen/native/Distributions.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// The native processing of n_sample==1 has poor MLU performance,
// so Dispatch-Stub is not used here.
void multinomial_with_replacement_mlu_kernel(
    at::Tensor& result,
    const at::Tensor& self,
    const int64_t n_sample,
    std::optional<at::Generator> gen) {
  auto self_contiguous = cnnl_contiguous(self);
  auto result_contiguous = cnnl_contiguous(result);
  cnnl_multinomial_internal(
      result_contiguous, self_contiguous, n_sample, true, gen);
  if (!result.is_same(result_contiguous)) {
    result.copy_(result_contiguous);
  }
}

constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);
at::Tensor& cnnl_multinomial_out(
    const at::Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    std::optional<at::Generator> gen,
    at::Tensor& result) {
  TORCH_CHECK(
      result.device() == self.device(),
      "multinomial arguments must have the same device");
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == at::ScalarType::Long,
      "multinomial expects Long tensor out, got: ",
      result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(
      with_replacement || (n_sample <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  if (result.numel() == 0) {
    return result;
  }
  // When n_sample==1, native uses multi-operators combination,
  // MLU uses large operators to improve performance.
  // if (!with_replacement || n_sample == 1) {
  if (!with_replacement) {
    // Sanity checks on `self`.
    auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
    TORCH_CHECK(
        is_valid.to<bool>(),
        "probability tensor contains either `inf`, `nan` or element < 0");
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool zero_prob_condition;
    if (self.dim() == 1) {
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    TORCH_CHECK(
        !zero_prob_condition,
        "invalid multinomial distribution (sum of probabilities <= 0)");
    Tensor q = at::empty_like(self).exponential_(1, gen);
    at::div_out(q, self, q);
    if (n_sample == 1) {
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, n_sample);
    }
    return result;
  }

  multinomial_with_replacement_mlu_kernel(result, self, n_sample, gen);
  return result;
}

at::Tensor cnnl_multinomial(
    const at::Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    std::optional<at::Generator> gen) {
  at::Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  cnnl_multinomial_out(self, n_sample, with_replacement, gen, result);
  return result;
}

} // namespace ops
} // namespace torch_mlu
