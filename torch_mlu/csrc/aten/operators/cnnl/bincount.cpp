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

#include "ATen/TensorUtils.h"
#include "ATen/Dispatch.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// Type list supported by GPU.
// coda path: aten/src/ATen/native/cuda/SummaryOps.cu
// input type: uint8_t, int8_t, int16, int32, int64
// weight type        convert type          output type
// at::kUndefined     NA                    at::kLong
// at::kFloat         NA                    at::kFloat
// others             at::kDouble           at::kDouble

at::Tensor cnnl_bincount(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weights_opt,
    int64_t minlength) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> weights_maybe_owned =
      at::borrow_from_optional_tensor(weights_opt);
  const at::Tensor& weights = *weights_maybe_owned;

  // TORCH_MLU is not support alertNotDeterministic now
  // globalContext().alertNotDeterministic("bincount_mlu");
  at::Tensor output;
  AT_DISPATCH_INTEGRAL_TYPES(self.scalar_type(), "bincount_mlu", [&] {
    TORCH_CHECK(minlength >= 0, "minlength should be >= 0");
    if (self.dim() == 1 && self.numel() == 0) {
      output = at::zeros(
          {minlength},
          at::kLong,
          c10::nullopt /* layout */,
          at::kPrivateUse1,
          c10::nullopt /* pin_memory */);
      return;
    }

    if (self.dim() != 1 ||
        (!std::is_same<scalar_t, uint8_t>::value &&
         *self.min().cpu().data_ptr<scalar_t>() < 0)) {
      AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
    }

    bool has_weights = weights.defined();
    if (has_weights &&
        (weights.dim() != 1 || weights.size(0) != self.size(0))) {
      AT_ERROR("weights should be 1-d and have the same length as input.");
    }

    const int64_t min_value = 0;
    const int64_t max_value =
        std::max(self.max().item<scalar_t>() + (int64_t)1, minlength);

    at::Tensor weight_convert;
    if (has_weights) {
      weight_convert = cnnl_contiguous(weights);
      // Check and convert weight type. And CNNL not support double.
      at::checkBackend(
          "MLU_tensor_histogram",
          {self, weight_convert},
          c10::Backend::PrivateUse1);
      const auto weight_scalar_type = weight_convert.scalar_type();
      // If weight type is double, TORCH_MLU don't need to cast.
      if (weight_scalar_type != at::kFloat &&
          weight_scalar_type != at::kDouble) {
        weight_convert =
            at::empty(weights.sizes(), weights.options().dtype(at::kDouble));
        cnnl_cast_internal(weights, weight_convert);
      }
      output = at::zeros({max_value}, weight_convert.options());
    } else {
      at::checkBackend(
          "MLU_tensor_histogram",
          {
              self,
          },
          c10::Backend::PrivateUse1);
      output = at::zeros(
          {max_value},
          at::kLong,
          c10::nullopt /* layout */,
          at::kPrivateUse1,
          c10::nullopt /* pin_memory */);
    }
    // Call internal function.
    at::Tensor self_contiguous = cnnl_contiguous(self);
    cnnl_bincount_internal(
        output, self_contiguous, weight_convert, min_value, max_value);
  });
  return output;
}

} // namespace ops
} // namespace torch_mlu
