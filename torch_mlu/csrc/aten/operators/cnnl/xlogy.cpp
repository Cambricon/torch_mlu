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
#include "aten/utils/binaryops_util.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/TensorIteratorBridge.h"
#include "aten/utils/types.h"

namespace torch_mlu {
namespace ops {

using at::native::xlogy_stub;

void xlogy_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0) {
    return;
  }
  auto input_x = iter.input(0);
  auto input_y = iter.input(1);
  auto output = iter.output(0);

  if (isCpuScalar(input_y)) {
    input_y = scalar_to_tensor_with_dtype(input_y, iter.common_dtype());
  }
  auto output_log = at::native::empty_like(input_y);
  cnnl_log_internal(output_log, input_y, CNNL_LOG_E);

  // Currently, cnnl not support xlogy so we implemented by operator
  // concatenation Hence we calculate x * log(y) first and then replace by nan/0
  // using where operator Refactor after cnnl supported
  if (isCpuScalar(input_x)) {
    cnnl_transform_out_internal(output, output_log, input_x.item(), 0);
    auto input_x_tensor =
        scalar_to_tensor_with_dtype(input_x, iter.common_dtype());
    cnnl_where_internal(output, input_x_tensor == 0, input_x_tensor, output);
  } else {
    cnnl_optensor_out_internal(
        output, output_log, input_x, 1, 1, 0, CNNL_OP_TENSOR_MUL);
    cnnl_where_internal(output, input_x == 0, input_x, output);
  }
  cnnl_where_internal(output, at::native::isnan(input_y), input_y, output);

  iter.cast_outputs();
}

at::Tensor cnnl_xlogy(const at::Tensor& self, const at::Scalar& other) {
  return at::native::xlogy(self, other);
}

at::Tensor cnnl_xlogy(const at::Scalar& self, const at::Tensor& other) {
  return at::native::xlogy(self, other);
}

at::Tensor& cnnl_xlogy_(at::Tensor& self, const at::Scalar& other) {
  return at::native::xlogy_(self, other);
}

REGISTER_PRIVATEUSE1_DISPATCH(xlogy_stub, &xlogy_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
