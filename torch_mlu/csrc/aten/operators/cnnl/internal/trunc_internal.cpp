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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_trunc_internal(at::Tensor& output, const at::Tensor& self) {
  if (self.numel() == 0) {
    return output;
  }
  TORCH_MLU_CHECK(
      at::isFloatingType(self.scalar_type()),
      "trunc input only support floating type");
  // get current handle
  auto handle = getCurrentHandle();
  auto self_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto descSelf = getTensorDesc(self_impl, CNNL_LAYOUT_ARRAY);
  auto descOutput = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

  // malloc memory
  auto self_ptr = self_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // final compute
  TORCH_CNNL_CHECK(cnnlTrunc(
      handle, descSelf.get(), self_ptr, descOutput.get(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
