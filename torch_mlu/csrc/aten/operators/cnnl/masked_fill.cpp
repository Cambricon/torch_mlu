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

#include "ATen/MemoryOverlap.h"
#include "ATen/TensorMeta.h"
#include "ATen/native/TensorAdvancedIndexing.h"
#include "ATen/native/IndexKernel.h"
#include "aten/TensorIteratorBridge.h"
#include "aten/DispatchStub.h"
#include "aten/utils/dispatch.h"
#include "aten/utils/types.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_masked_fill_kernel(
    at::TensorIterator& iter,
    at::Tensor value,
    const at::Tensor& mask) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_masked_fill_internal(output, self, mask, value);
}

at::Tensor& cnnl_masked_fill_(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Scalar& value) {
  auto value_tensor = at::native::scalar_tensor(
      value, self.scalar_type(), c10::nullopt, at::kCPU, c10::nullopt);
  return cnnl_masked_fill_(self, mask, value_tensor);
}

at::Tensor& cnnl_masked_fill_(
    at::Tensor& self,
    const at::Tensor& mask,
    const at::Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0,
      "masked_fill_ only supports a 0-dimensional ",
      "value tensor, but got tensor ",
      "with ",
      value.dim(),
      " dimension(s).");
  TORCH_CHECK(
      value.scalar_type() != at::ScalarType::Byte,
      "value tensor does not support uint8");
  // We hit this function if either of the input tensor lives on MLU.
  // It is ok, if `value` is `CPU` tensor but we should not allow `self` or
  // `mask` to be CPU tensor. Check for `self` and `mask` being on same device
  // exists in `masked_fill__cuda` (Scalar version).
  TORCH_CHECK(
      !self.device().is_cpu(),
      "masked_fill_: Expected inputs to be on same device")
  TORCH_CHECK(
      self.device() == mask.device(),
      "expected self and mask to be on the same device, ",
      "but got mask on ",
      mask.device(),
      " and self on ",
      self.device());
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool,
      "masked_fill only supports boolean masks, but got dtype ",
      mask.scalar_type());
  auto maybe_outnames =
      at::namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {
    TORCH_WARN(
        "Use of masked_fill_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");
  auto iter = at::TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_output(self)
                  .add_input(self)
                  .add_input(*b_mask)
                  .build();

  // b_mask is just used for TensorIterator check, we use mask directly
  // because cnnlMasked_v4 can support expand internally
  // and can achieve better performance.
  cnnl_masked_fill_kernel(iter, value, mask);
  at::namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}
} // namespace ops
} // namespace torch_mlu
