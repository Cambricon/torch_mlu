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

#include "ATen/native/UnaryOps.h"

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {
namespace ops {
using at::native::nan_to_num_stub;
using at::native::nan_to_num_stub_DECLARE_DISPATCH_type;

Tensor& cnnl_nan_to_num_out(
    const Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf,
    Tensor& result) {
  return at::native::nan_to_num_out(self, nan, pos_inf, neg_inf, result);
}

void nan_to_num_mlu_kernel(
    at::TensorIteratorBase& iter,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  if (iter.numel() == 0) {
    return;
  }
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "nan_to_num");
  auto output = iter.output(0);
  auto input = iter.input(0);
  cnnl_nan_to_num_internal(output, input, nan, pos_inf, neg_inf);
  iter_bridge.cast_outputs(iter);
}

REGISTER_PRIVATEUSE1_DISPATCH(nan_to_num_stub, &nan_to_num_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
