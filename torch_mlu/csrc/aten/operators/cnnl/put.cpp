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
#include "aten/utils/dispatch.h"

using at::native::put_stub;
using at::native::put_stub_DECLARE_DISPATCH_type;

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_put_(
    at::Tensor& self,
    const at::Tensor& index,
    const at::Tensor& source,
    const bool accumulate) {
  return at::native::put_(self, index, source, accumulate);
}

void put_mlu_kernel(
    at::TensorIterator& iter,
    const at::TensorBase& self,
    const bool accumulate) {
  auto self_tensor = Tensor(self);
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self_tensor.scalar_type(),
      "put",
      [&] {
        auto numel = self_tensor.numel();
        auto source = iter.input(0);
        auto index = iter.input(1);
        auto idx_temp = index.clone();
        auto indices = idx_temp.reshape({
            index.numel(),
        });
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
        auto input = self_tensor.reshape({
            numel,
        });
        auto source_reshaped = source.reshape({
            source.numel(),
        });
        auto input_contiguous = cast_long_to_int_if_needed(
            cnnl_contiguous(input, c10::MemoryFormat::Contiguous));
        auto indices_contiguous =
            cnnl_contiguous(indices, c10::MemoryFormat::Contiguous);
        auto source_contiguous = cast_long_to_int_if_needed(
            cnnl_contiguous(source_reshaped, c10::MemoryFormat::Contiguous));

        // TODO(chentianyi1): reuse the index_put_internal.
        std::vector<at::Tensor> idx_vec;
        idx_vec.push_back(indices_contiguous);
        cnnl_index_put_internal(
            input_contiguous,
            input_contiguous,
            idx_vec,
            source_contiguous,
            accumulate);
        if (self_tensor.data_ptr() != input_contiguous.data_ptr()) {
          self_tensor.copy_(input_contiguous.reshape(self.sizes()));
        }
      });
}
REGISTER_PRIVATEUSE1_DISPATCH(put_stub, &put_mlu_kernel);
} // namespace ops
} // namespace torch_mlu
