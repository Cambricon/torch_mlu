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
#include "ATen/native/Fill.h"
#include "ATen/TensorMeta.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/binaryops_util.h"
#include "aten/DispatchStub.h"
#include "aten/utils/dispatch.h"
#include "aten/utils/types.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_fill_(at::Tensor& self, const at::Tensor& other) {
  TORCH_MLU_CHECK(
      other.dim() == 0,
      "fill_ only supports 0-dimension value tensor but got tensor with ",
      other.dim(),
      " dimensions.");
  if (self.numel() == 0)
    return self;
  // TODO(CNNLCORE-15618) Remove this tmp arg after cnnlCastDataType supports
  // complex inputs
  at::Tensor other_ = other;
  if (at::isComplexType(self.scalar_type()))
    other_ = other.to(at::kCPU);
  auto value = other_.to(self.scalar_type());
  cnnl_fill_internal(self, value);
  return self;
}

at::Tensor& cnnl_fill_(at::Tensor& self, const at::Scalar& other) {
  if (self.numel() == 0)
    return self;
  return cnnl_fill_internal(self, other);
}
// Type list supported by fill_kernel_cuda
// cuda path: aten/src/ATen/native/cuda/FillKernel.cu
// iter.common_dtype()
// at::ScalarType::Bool
// at::ScalarType::Byte
// at::ScalarType::Char
// at::ScalarType::Int
// at::ScalarType::Long
// at::ScalarType::Short
// at::ScalarType::Float
// at::ScalarType::Double
// at::ScalarType::ComplexFloat
// at::ScalarType::ComplexDouble
// at::ScalarType::BFloat16
// at::ScalarType::Half
// at::ScalarType::ComplexHalf

// Type list supported by CNNLFill
// iter.common_dtype()
// at::ScalarType::Bool
// at::ScalarType::Byte
// at::ScalarType::Char
// at::ScalarType::Int
// at::ScalarType::Long
// at::ScalarType::Short
// at::ScalarType::Float
// at::ScalarType::Double
// at::ScalarType::Half
// at::ScalarType::Bfloat16
// at::ScalarType::ComplexFloat
// at::ScalarType::ComplexDouble
// at::ScalarType::ComplexHalf

} // namespace ops
} // namespace torch_mlu
