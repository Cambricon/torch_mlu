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
#include "ATen/native/Lerp.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {
using namespace at::native;

// out = start + weight * (end - start)
void lerp_tensor_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto start = iter.input(0);
  auto end = iter.input(1);
  auto weight = iter.input(2);
  cnnl_lerp_internal(output, start, end, weight);
  iter.cast_outputs();
}

void lerp_scalar_mlu_kernel(
    at::TensorIteratorBase& iter,
    const c10::Scalar& weight) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto start = iter.input(0);
  auto end = iter.input(1);
  auto weight_tensor = wrapped_scalar_tensor(weight);
  auto weight_device_tensor =
      scalar_to_tensor_with_dtype(weight_tensor, output.scalar_type());
  cnnl_lerp_internal(output, start, end, weight_device_tensor);
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(
    lerp_kernel_scalar_weight,
    &lerp_scalar_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    lerp_kernel_tensor_weight,
    &lerp_tensor_mlu_kernel);
} // namespace ops
} // namespace torch_mlu
