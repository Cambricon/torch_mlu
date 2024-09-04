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

at::Tensor cnnl_binary_cross_entropy_with_logits(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction) {
  at::Tensor output;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "cnnl_binary_cross_entropy_with_logits",
      [&] {
        auto memory_format = self.suggest_memory_format();
        auto self_contiguous = cnnl_contiguous(self, memory_format);
        auto target_contiguous = cnnl_contiguous(target, memory_format);

        const Tensor& weight = *at::borrow_from_optional_tensor(weight_opt);
        at::Tensor weight_contiguous =
            weight.defined() ? cnnl_contiguous(weight, memory_format) : weight;

        const Tensor& pos_weight =
            *at::borrow_from_optional_tensor(pos_weight_opt);
        at::Tensor pos_weight_contiguous = pos_weight.defined()
            ? cnnl_contiguous(pos_weight, memory_format)
            : pos_weight;

        cnnl_bce_with_logits_internal(
            self_contiguous,
            target_contiguous,
            weight_contiguous,
            pos_weight_contiguous,
            reduction,
            output);
      });
  return output;
}

} // namespace ops
} // namespace torch_mlu
