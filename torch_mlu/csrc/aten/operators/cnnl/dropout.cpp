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

#include "ATen/native/TensorIterator.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {
namespace ops {

inline std::tuple<at::Tensor, at::Tensor> _fused_dropout_impl(
    const at::Tensor& self,
    double p,
    std::optional<at::Generator> generator,
    bool mask_is_bool) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "cnnl__fused_dropout",
      [&]() {
        auto memory_format = self.suggest_memory_format();
        auto input_contiguous = cnnl_contiguous(self, memory_format);
        auto output = at::empty_like(input_contiguous, memory_format);

        // create mask tensor
        auto mask_dtype = mask_is_bool ? at::kBool : at::kByte;
        at::Tensor mask = at::empty(
            input_contiguous.sizes(),
            input_contiguous.options().dtype(mask_dtype),
            memory_format);

        size_t self_num = static_cast<size_t>(input_contiguous.numel());
        if (self_num == 0) {
          return std::tuple<at::Tensor, at::Tensor>(
              input_contiguous.clone(), mask);
        }
        fused_dropout_internal(output, mask, input_contiguous, p, generator);
        return std::tuple<at::Tensor, at::Tensor>(output, mask);
      });
}

inline at::Tensor _masked_scale_impl(
    const at::Tensor& self,
    const at::Tensor& mask,
    double scale) {
  // Build TensorIterator.
  at::Tensor output = at::empty_like(self, self.suggest_memory_format());
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(output)
                  .add_input(self)
                  .add_input(mask)
                  .build();

  // Build MLU TensorIterator bridge.
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "_masked_scale");

  // Based on cuda kernel function, cast scale to float based on mlu support
  // input dtype. aten/src/ATen/AccumulateType.h
  // aten/src/ATen/native/cuda/Dropout.cu
  float scale_f = static_cast<float>(scale);
  auto output_tmp = iter.output(0);

  cnnl_masked_scale_internal(output_tmp, iter.input(0), iter.input(1), scale_f);
  iter_bridge.cast_outputs(iter);
  return output;
}

std::tuple<at::Tensor, at::Tensor> cnnl__fused_dropout(
    const at::Tensor& self,
    double p,
    std::optional<at::Generator> generator) {
  return _fused_dropout_impl(self, p, generator, false);
}

at::Tensor cnnl__masked_scale(
    const at::Tensor& self,
    const at::Tensor& mask,
    double scale) {
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Byte,
      "mask should be torch.uint8 dtype");
  return _masked_scale_impl(self, mask, scale);
}

std::tuple<at::Tensor, at::Tensor> cnnl_native_dropout(
    const at::Tensor& input,
    double p,
    std::optional<bool> train) {
  // short-cut for train == false
  auto dtype_trans = c10::CppTypeToScalarType<bool>::value;
  if (train.has_value() && !train.value()) {
    return std::make_tuple(
        input.clone(),
        at::ones_like(input, input.options().dtype(dtype_trans)));
  }
  // short-cut
  if (p == 1) {
    auto ret = at::zeros_like(input);
    auto mask = at::zeros_like(input, input.options().dtype(dtype_trans));
    return std::tuple<at::Tensor, at::Tensor>(ret, mask);
  }
  double p1m = 1. - p;
  return _fused_dropout_impl(input, p1m, c10::nullopt, true);
}

at::Tensor cnnl_native_dropout_backward(
    const at::Tensor& grad_output,
    const at::Tensor& mask,
    double scale) {
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool,
      "Mask should be Bool Scalar Type",
      mask.scalar_type());
  return _masked_scale_impl(grad_output, mask, scale);
}

} // namespace ops
} // namespace torch_mlu
