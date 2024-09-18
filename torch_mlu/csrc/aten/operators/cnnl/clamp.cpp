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
#include "ATen/native/TensorCompare.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {
// min/max tensor type will dispatch into maximum_stub and minimun_stub.
// more details:
// https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/native/TensorCompare.cpp#L634-L635
void inline launch_clamp_scalar_mlu(
    at::TensorIteratorBase& iter,
    at::optional<at::Scalar> min,
    at::optional<at::Scalar> max) {
  auto self = iter.input(0);
  auto output = iter.output(0);
  cnnl_clamp_internal(output, self, min, max);
  iter.cast_outputs();
}

void clamp_min_scalar_mlu_kernel(at::TensorIteratorBase& iter, at::Scalar min) {
  launch_clamp_scalar_mlu(iter, min, at::optional<at::Scalar>());
}

void clamp_max_scalar_mlu_kernel(at::TensorIteratorBase& iter, at::Scalar max) {
  launch_clamp_scalar_mlu(iter, at::optional<at::Scalar>(), max);
}

void clamp_scalar_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& min,
    const at::Scalar& max) {
  launch_clamp_scalar_mlu(iter, min, max);
}

void clamp_mlu_kernel(at::TensorIteratorBase& iter) {
  auto self = iter.input(0);
  auto min = iter.input(1);
  auto max = iter.input(2);
  auto output = iter.output(0);
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "clamp_mlu",
      [&] { cnnl_clamp_tensor_internal(output, self, min, max); });
  iter.cast_outputs();
}

using namespace at::native;
REGISTER_PRIVATEUSE1_DISPATCH(
    clamp_min_scalar_stub,
    &clamp_min_scalar_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    clamp_max_scalar_stub,
    &clamp_max_scalar_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(clamp_scalar_stub, &clamp_scalar_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(clamp_stub, &clamp_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
