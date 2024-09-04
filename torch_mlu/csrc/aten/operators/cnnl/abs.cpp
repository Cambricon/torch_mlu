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

#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_abs_out(const at::Tensor& self, at::Tensor& result) {
  // Since cnnl kernel can handle the complex-in-float-out case directly,
  // so don't use the stub to dispatch
  if (self.is_complex() && !result.is_complex()) {
    // Checks if the corresponding float type can be cast to the desired dtype
    const auto float_type = c10::toRealValueType(self.scalar_type());
    TORCH_CHECK(
        canCast(float_type, result.scalar_type()),
        "result type ",
        float_type,
        " can't be cast to the desired output type ",
        result.scalar_type());

    // Runs the function complex->complex, as TensorIterator expects
    Tensor complex_result = at::empty({0}, self.options());
    auto iter = at::TensorIterator::unary_op(complex_result, self);
    TensorIteratorBridge iter_bridge;
    iter_bridge.to_build(iter, "abs");

    // Gets the actual result and returns it
    at::native::resize_output(result, iter.output(0).sizes());
    if (result.strides() != iter.output(0).strides()) {
      Tensor result_empty = cnnl_empty_strided(
          iter.output(0).sizes(),
          iter.output(0).strides(),
          result.scalar_type(),
          result.layout(),
          result.device());
      cnnl_abs_internal(result_empty, iter.input(0));
      result.copy_(result_empty);
    } else {
      cnnl_abs_internal(result, iter.input(0));
    }
    return result;
  }

  auto iter = at::TensorIterator::unary_op(result, self);
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "abs");
  auto output = create_int_tensor_if_needed(iter.output(0));
  auto input = cast_long_to_int_if_needed(iter.input(0));
  cnnl_abs_internal(output, input);
  cast_int_to_long_if_needed(output, iter.output(0));
  iter_bridge.cast_outputs(iter);
  return result;
}

} // namespace ops
} // namespace torch_mlu
