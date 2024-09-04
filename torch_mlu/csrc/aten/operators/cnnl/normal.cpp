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
#include "ATen/Generator.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/utils/dispatch.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {
namespace ops {

void normal_mlu_kernel(
    const at::TensorBase& self,
    double mean,
    double std,
    std::optional<at::Generator> gen) {
  TORCH_CHECK(
      c10::isFloatingType(self.scalar_type()),
      "self dtype of mlu normal op not implemented for ",
      self.scalar_type());

  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "default");
  auto output = iter.output(0);
  cnnl_normal_internal(output, mean, std, gen);
  iter_bridge.cast_outputs(iter);
}

at::Tensor& cnnl_normal_(
    at::Tensor& self,
    double mean,
    double std,
    std::optional<at::Generator> generator) {
  return at::native::normal_(self, mean, std, generator);
}

at::Tensor cnnl_normal(
    const at::Tensor& mean,
    double std,
    std::optional<at::Generator> generator) {
  return at::native::normal(mean, std, generator);
}

at::Tensor& cnnl_normal_out(
    const at::Tensor& mean,
    double std,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  return at::native::normal_out(mean, std, generator, out);
}

at::Tensor cnnl_normal(
    double mean,
    const at::Tensor& std,
    std::optional<at::Generator> generator) {
  return at::native::normal(mean, std, generator);
}

at::Tensor& cnnl_normal_out(
    double mean,
    const at::Tensor& std,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  return at::native::normal_out(mean, std, generator, out);
}

at::Tensor cnnl_normal(
    const at::Tensor& mean,
    const at::Tensor& std,
    std::optional<at::Generator> generator) {
  return at::native::normal(mean, std, generator);
}

at::Tensor& cnnl_normal_out(
    const at::Tensor& mean,
    const at::Tensor& std,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  return at::native::normal_out(mean, std, generator, out);
}

using namespace at::native;
REGISTER_PRIVATEUSE1_DISPATCH(normal_stub, &normal_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
