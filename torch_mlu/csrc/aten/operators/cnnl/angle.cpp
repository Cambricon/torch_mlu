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

// Pytorch code path:
// aten/src/ATen/native/cuda/UnaryComplexKernels.cu
// List of data types supported by GPU:
// at::ScalarType::Byte, at::ScalarType::Char, at::ScalarType::Int,
// at::ScalarType::Long, at::ScalarType::Short, at::ScalarType::Float,
// at::ScalarType::Double, at::ScalarType::ComplexFloat,
// at::ScalarType::ComplexDouble, at::ScalarType::ComplexHalf

// List of data types supported by MLU:
// at::ScalarType::Byte, at::ScalarType::Char, at::ScalarType::Int,
// at::ScalarType::Long, at::ScalarType::Short, at::ScalarType::Float,
// at::ScalarType::Double,at::ScalarType::ComplexFloat,
// at::ScalarType::ComplexDouble, other types raise error: "not implemented for
// 'MLU'".

// Input type and output type rule:
// 1) If input type is integral type, return type is float.
// 2) If input type is float type, return type is same as input type.
// 3) If input type is complex type, and output tensor is not defined, then
//    the return type is real value type of input type.
// 4) If input and output type is complex type, the return type is same as
// output type.

namespace torch_mlu {
namespace ops {

using at::native::angle_stub;
using at::native::angle_stub_DECLARE_DISPATCH_type;

at::Tensor cnnl_angle(const at::Tensor& input) {
  return at::native::angle(input);
}

at::Tensor& cnnl_angle_out(const at::Tensor& self, at::Tensor& result) {
  return at::native::angle_out(self, result);
}

void angle_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0) {
    return;
  }
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "angle");
  auto output = iter.output(0);
  cnnl_angle_internal(output, iter.input(0));
  iter_bridge.cast_outputs(iter);
}

REGISTER_PRIVATEUSE1_DISPATCH(angle_stub, &angle_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
