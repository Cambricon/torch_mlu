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

#include "ATen/core/TensorBody.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {
at::Tensor cnnl_repeat_interleave_internal(
    at::Tensor& output,
    const at::Tensor& index,
    const at::Tensor& repeats) {
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor index_desc;
  CnnlTensorDescriptor output_desc;
  CnnlTensorDescriptor repeat_desc;
  index_desc.set(index);
  output_desc.set(output);
  repeat_desc.set(repeats);

  auto index_impl = getMluTensorImpl(index);
  auto output_impl = getMluTensorImpl(output);
  auto repeat_impl = getMluTensorImpl(repeats);
  auto index_ptr = index_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto repeat_ptr = repeat_impl->mlu_data_ptr();

  TORCH_CNNL_CHECK(cnnlRepeatInterleave(
      handle,
      index_desc.desc(),
      index_ptr,
      repeat_desc.desc(),
      repeat_ptr,
      0,
      output_desc.desc(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
