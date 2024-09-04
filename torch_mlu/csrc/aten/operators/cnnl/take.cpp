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

#include "ATen/native/TensorIterator.h"
#include "ATen/NativeFunctions.h"
#include "ATen/native/IndexKernel.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "ATen/native/TensorAdvancedIndexing.h"
#include "aten/utils/dispatch.h"

using at::native::take_stub;
using at::native::take_stub_DECLARE_DISPATCH_type;

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_take(const at::Tensor& self, const at::Tensor& index) {
  return at::native::take(self, index);
}

at::Tensor& cnnl_take_out(
    const at::Tensor& self,
    const at::Tensor& index,
    at::Tensor& out) {
  return at::native::take_out(self, index, out);
}

void take_mlu_kernel(at::TensorIterator& iter, const at::TensorBase& self) {
  auto self_tensor = Tensor(self);
  AT_DISPATCH_MLU_TENSOR_SCLAER_TYPES(iter.dtype(), "take", [&] {
    auto out = iter.output(0);
    auto numel = self_tensor.numel();
    auto index = iter.input(0);
    // TODO(CNNLCORE-13920): remove the following for loop after cnnlIndexSelect
    // support boundary checks for indices
    auto indices = index
                       .reshape({
                           index.numel(),
                       })
                       .to(at::Device(at::kCPU));
    for (int64_t i = 0; i < index.numel(); i++) {
      auto idx = indices[i].item<int64_t>();
      TORCH_CHECK(
          idx < numel && idx >= -numel,
          "out of range: tried to access index ",
          idx,
          " on a tensor of ",
          numel,
          " elements.");
      if (idx < 0) {
        idx += numel;
      }
      indices[i] = idx;
    }
    indices = indices.to(at::Device(at::kPrivateUse1));
    auto input = self_tensor.reshape({
        self.numel(),
    });
    auto output_reshaped = out.reshape({
        out.numel(),
    });
    auto input_contiguous =
        cnnl_contiguous(input, c10::MemoryFormat::Contiguous);
    auto indices_contiguous =
        cnnl_contiguous(indices, c10::MemoryFormat::Contiguous);
    auto output_contiguous =
        cnnl_contiguous(output_reshaped, c10::MemoryFormat::Contiguous);
    cnnl_take_internal(input_contiguous, indices_contiguous, output_contiguous);
    if (out.data_ptr() != output_contiguous.data_ptr()) {
      out.copy_(output_contiguous.reshape(out.sizes()));
    }
  });
}
REGISTER_PRIVATEUSE1_DISPATCH(take_stub, &take_mlu_kernel);
} // namespace ops
} // namespace torch_mlu
