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

void cnnl_slice_backward_internal(
    const at::Tensor& grad_output,
    const std::vector<int>& begins,
    const std::vector<int>& ends,
    const std::vector<int>& strides,
    at::Tensor& grad_input) {
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_input_impl = getMluTensorImpl(grad_input);

  auto grad_output_desc = getTensorDesc(grad_output_impl);
  auto grad_input_desc = getTensorDesc(grad_input_impl);

  auto grad_output_ptr = grad_output_impl->mlu_data_ptr();
  auto grad_input_ptr = grad_input_impl->mlu_data_ptr();

  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlStridedSliceBackward(
      handle,
      begins.data(),
      ends.data(),
      strides.data(),
      grad_output_desc.get(),
      grad_output_ptr,
      grad_input_desc.get(),
      grad_input_ptr));
}

} // namespace ops
} // namespace torch_mlu