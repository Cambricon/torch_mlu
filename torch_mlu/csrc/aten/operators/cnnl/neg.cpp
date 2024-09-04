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
#include "aten/operators/cnnl/copy_utils.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

using at::native::neg_stub;
using at::native::neg_stub_DECLARE_DISPATCH_type;

// Pytorch neg op code path:
// aten/src/ATen/native/cuda/UnarySignKernels.cu
// List of data types supported by GPU:
// at::ScalarType::Byte, at::ScalarType::Char, at::ScalarType::Int,
// at::ScalarType::Long, at::ScalarType::Short, at::ScalarType::Float,
// at::ScalarType::Double, at::ScalarType::ComplexFloat,
// at::ScalarType::ComplexDouble, at::ScalarType::BFloat16,
// at::ScalarType::Half, at::ScalarType::ComplexHalf

// List of data types supported by MLU:
// at::ScalarType::Int, at::ScalarType::Float, at::ScalarType::Half
// at::ScalarType::Double, at::ScalarType::Long.
// other types raise error: "not implemented for 'MLU'".
void neg_mlu_kernel(at::TensorIteratorBase& iter) {
  // gpu 0-element check is in:
  // https://github.com/pytorch/pytorch/blob/release/1.13/aten/src/ATen/native/cuda/Loops.cuh#L86
  if (iter.numel() == 0) {
    return;
  }
  auto output = create_int_tensor_if_needed(iter.output(0));
  auto input = cast_long_to_int_if_needed(iter.input(0));
  cnnl_neg_internal(output, input);
  cast_int_to_long_if_needed(output, iter.output(0));
}

REGISTER_PRIVATEUSE1_DISPATCH(neg_stub, &neg_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
