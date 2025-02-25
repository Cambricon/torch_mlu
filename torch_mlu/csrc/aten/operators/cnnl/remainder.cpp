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

#include <ATen/native/BinaryOps.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/TensorIteratorBridge.h"
#include "aten/DispatchStub.h"
#include "aten/operators/cnnl/cnnlOpParams.h"
#include "aten/utils/binaryops_util.h"

namespace torch_mlu {
namespace ops {

using at::native::remainder_stub;

at::Tensor cnnl_remainder(const at::Scalar& self, const at::Tensor& other) {
  return at::native::remainder(self, other);
}

void remainder_kernel_mlu(at::TensorIteratorBase& iter) {
  auto output = create_int_tensor_if_needed(iter.output(0));
  auto self = iter.input(0);
  auto other = iter.input(1);
  auto self_tensor = cast_long_to_int_if_needed(
      scalar_to_tensor_with_dtype(self, output.scalar_type()));
  auto other_tensor = cast_long_to_int_if_needed(
      scalar_to_tensor_with_dtype(other, output.scalar_type()));
  if (self_tensor.numel() == 0)
    return;
  cnnl_remainder_internal(output, self_tensor, other_tensor);
  cast_int_to_long_if_needed(output, iter.output(0));
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(remainder_stub, &remainder_kernel_mlu);

} // namespace ops
} // namespace torch_mlu
