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
  auto output = iter_bridge.output(iter, 0);
  cnnl_angle_internal(output, iter_bridge.input(iter, 0));
  // Pytorch unary_op_impl_float_out function calls cast_outputs function.
  // This may be a bug of Pytorch, because the cast_outputs function is
  // only for cpu kernels, and only once appear in unary_op_impl_float_out
  // function. Now we add cast_outputs function here. 1) MLU little ops will
  // call this function twice when using pytorch unary_op_impl_float_out. This
  // is ok, because the second call of cast_outputs function will do nothing,
  // and the first call of cast_outputs function is copy tensor to original
  // tensor and then move original tensor to tensor. When second call of
  // cast_outputs function, original tensor is undefined and return. 2) For
  // other branches, we need this function to copy tensor to original tensor,
  // and then move original tensor to tensor. Like unary_op_impl_float.
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(angle_stub, &angle_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
