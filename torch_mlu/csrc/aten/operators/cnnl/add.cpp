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
#include "ATen/TensorMeta.h"
#include "ATen/OpMathType.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/binaryops_util.h"
#include "aten/DispatchStub.h"
#include "aten/utils/dispatch.h"
#include "aten/utils/accumulate_type.h"

namespace torch_mlu {
namespace ops {

using at::native::add_stub;

at::Tensor cnnl_add(
    const at::Tensor& input,
    const at::Scalar& other,
    const at::Scalar& alpha_scalar) {
  return at::native::add(input, other, alpha_scalar);
}

at::Tensor& cnnl_add_(
    at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha) {
  return at::native::add_(self, other, alpha);
}

// Type list supported by GPU.
// coda path: build/aten/src/ATen/UfuncCUDA_add.cu
// iter.common_dtype()               compute cpp type      alpha cpp type
// at::ScalarType::Bool              bool                   bool
// at::ScalarType::Byte              uint8_t                uint8_t
// at::ScalarType::Char              int8_t                 int8_t
// at::ScalarType::Int               int32                  int32
// at::ScalarType::Long              int64                  int64
// at::ScalarType::Short             int16                  int16
// at::ScalarType::Float             float32                float32
// at::ScalarType::Double            float64                float64
// at::ScalarType::ComplexFloat      c10::complex<float>    c10::complex<float>
// at::ScalarType::ComplexDouble     c10::complex<double>   c10::complex<double>
// at::ScalarType::BFloat16          float32                float32
// at::ScalarType::Half              float32                float32
// at::ScalarType::ComplexHalf       c10::complex<float>    c10::complex<float>
// compute cpp type mean operator compute type, and self tensor type and other
// tensor type dynamic cast to compute type in gpu operator kernel.
// https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/ufunc/add.h#L15

// Type list supported by CNNL OpTensor.
// at::ScalarType::Half, at::ScalarType::Float, at::ScalarType::Int.
// CATCH implicit convert tensor type to CNNL type by using internal tensor.
// iter.common_dtype()               compute cpp type      alpha cpp type
// at::ScalarType::Bool              int32                  int32
// at::ScalarType::Byte              int32                  int32
// at::ScalarType::Char              int32                  int32
// at::ScalarType::Int               int32                  int32
// at::ScalarType::Long              int32                  int32
// at::ScalarType::Short             int32                  int32
// at::ScalarType::Float             float32                float32
// at::ScalarType::Double            float32                float32
// at::ScalarType::Half              float32                float32
// at::ScalarType::BFloat16          float32                float32

// PYTORCH side add:       output = input + alpha * other
// CATCH side OpTensor:    c = op(alpha1[0] * a, alpha2[0] * b) + beta[0] * c
// CATCH side Transformer: output = alpha * input + beta (input and output type
// need be same) OpTensor alpha1 and alpha2 type is float when input and other
// type is not int, otherwise is int. CATCH side transformer: output = alpha *
// input + beta transformer alpha and beta type is float when input and other
// type is not int, otherwise is int. CNNL OpTensor and transformer only support
// int, half and float type.
void cnnl_add_kernel(at::TensorIteratorBase& iter, const at::Scalar& scalar) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  auto other = iter.input(1);
  // For add support mixed input types, so maybe self is int, other is float,
  // and output is float.
  if (isCpuScalar(other) && (self.scalar_type() == output.scalar_type())) {
    AT_DISPATCH_MLU_FLOAT_HALF_INT_COMPLEX_AND_BFLOAT16(
        output.scalar_type(), "transform add", [&] {
          using opmath_t = MLUOpMathType_t<scalar_t>;
          opmath_t other_data =
              iter.scalar_value<opmath_t>(2) * (scalar).to<opmath_t>();
          cnnl_transform_out_internal(output, self, 1, other_data);
        });
  } else if (
      isCpuScalar(self) && (other.scalar_type() == output.scalar_type())) {
    cnnl_transform_out_internal(output, other, scalar, self.item());
  } else {
    cnnl_optensor_out_with_scalar_internal(
        output, self, other, 1, scalar, 0, CNNL_OP_TENSOR_ADD);
  }
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(add_stub, &cnnl_add_kernel);

TORCH_IMPL_FUNC(add_out_mlu)
(const at::Tensor& self,
 const at::Tensor& other,
 const at::Scalar& alpha,
 const at::Tensor& result) {
  cnnl_add_kernel(*this, alpha);
}

} // namespace ops
} // namespace torch_mlu
