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

#include "ATen/native/BinaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

using at::Tensor;
using at::native::DispatchStub;
using at::native::div_floor_stub;
using at::native::div_true_stub;
using at::native::div_trunc_stub;

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_div(const at::Tensor& self, const at::Scalar& other) {
  return at::native::div(self, other);
}

at::Tensor& cnnl_div_(at::Tensor& self, const at::Scalar& other) {
  return at::native::div_(self, other);
}

void div_trunc_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  auto other = iter.input(1);
  if (isCpuScalar(other) && at::isFloatingType(output.scalar_type())) {
    // (TODO)shangang: Using transform and trunc to instead div, cause half
    // tensor + float scalar will using half to caculate, that maybe cause
    // percision loss. cnnl kernel need to support scalar tensor in future.
    cnnl_transform_out_internal(
        output, self, 1.0 / (other.item().to<float>()), 0);
    cnnl_trunc_internal(output, output);
  } else {
    TORCH_CHECK(
        at::isFloatingType(iter.common_dtype()) ||
            at::isIntegralType(iter.common_dtype(), /*includeBool=*/false),
        "div trunc inputs only support floating/integral type");
    auto self_tensor = scalar_to_tensor_with_dtype(self, iter.common_dtype());
    auto other_tensor = scalar_to_tensor_with_dtype(other, iter.common_dtype());
    cnnl_div_out_internal(output, self_tensor, other_tensor, "trunc");
  }
  iter.cast_outputs();
}

void div_floor_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  auto other = iter.input(1);
  if (isCpuScalar(other) && at::isFloatingType(output.scalar_type())) {
    // (TODO)shangang: Using transform and trunc to instead div, cause half
    // tensor + float scalar will using half to caculate, that maybe cause
    // percision loss. cnnl kernel need to support scalar tensor in future.
    cnnl_transform_out_internal(
        output, self, 1.0 / (other.item().to<float>()), 0);
    cnnl_floor_internal(output, output);
  } else {
    TORCH_CHECK(
        at::isFloatingType(iter.common_dtype()) ||
            at::isIntegralType(iter.common_dtype(), /*includeBool=*/false),
        "div floor inputs only support floating/integral type");
    auto self_tensor = scalar_to_tensor_with_dtype(self, iter.common_dtype());
    auto other_tensor = scalar_to_tensor_with_dtype(other, iter.common_dtype());
    cnnl_div_out_internal(output, self_tensor, other_tensor, "floor");
  }
  iter.cast_outputs();
}

void div_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  auto other = iter.input(1);
  if (isCpuScalar(other)) {
    cnnl_transform_out_internal(
        output, self, 1.0 / (other.item().to<float>()), 0);
  } else {
    auto self_tensor = scalar_to_tensor_with_dtype(self, iter.common_dtype());
    cnnl_div_out_internal(output, self_tensor, other, "true");
  }
  iter.cast_outputs();
}

at::Tensor& cnnl_true_divide_(at::Tensor& self, const at::Tensor& other) {
  return at::native::true_divide_(self, other);
}

REGISTER_PRIVATEUSE1_DISPATCH(div_floor_stub, &div_floor_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(div_trunc_stub, &div_trunc_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(div_true_stub, &div_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
