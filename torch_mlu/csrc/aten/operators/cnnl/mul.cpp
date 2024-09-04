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
#include "aten/utils/binaryops_util.h"
#include "aten/DispatchStub.h"
#include "aten/utils/dispatch.h"
#include "aten/utils/types.h"

namespace torch_mlu {
namespace ops {

using at::native::mul_stub;

// Type list is same with add, and using mul_stub in impl function.
// https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/cuda/BinaryMulKernel.cu#L42

at::Tensor cnnl_mul(const at::Tensor& self, const at::Scalar& other) {
  return at::native::mul(self, other);
}

at::Tensor& cnnl_mul_(at::Tensor& self, const at::Scalar& other) {
  return at::native::mul_(self, other);
}

// PYTORCH side mul:       output = input * other
// TORCH_MLU side OpTensor:    c = op(alpha1[0] * a, alpha2[0] * b) + beta[0] *
// c TORCH_MLU side Transformer: output = alpha * input + beta (input and output
// type need be same) OpTensor alpha1 and alpha2 type is float when input and
// other type is not int, otherwise is int. TORCH_MLU side transformer: output =
// alpha * input + beta transformer alpha and beta type is float when input and
// other type is not int, otherwise is int. CNNL OpTensor and transformer only
// support int, half and float type.
void mul_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  auto other = iter.input(1);
  if (isCpuScalar(other) && (output.scalar_type() == self.scalar_type())) {
    cnnl_transform_out_internal(output, self, other.item(), 0);
  } else if (
      isCpuScalar(self) && (output.scalar_type() == other.scalar_type())) {
    cnnl_transform_out_internal(output, other, self.item(), 0);
  } else {
    cnnl_optensor_out_with_scalar_internal(
        output, self, other, 1, 1, 0, CNNL_OP_TENSOR_MUL);
  }
  iter.cast_outputs();
}
REGISTER_PRIVATEUSE1_DISPATCH(mul_stub, &mul_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
