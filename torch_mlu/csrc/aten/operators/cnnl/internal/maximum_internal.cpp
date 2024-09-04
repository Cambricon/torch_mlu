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
#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/binaryops_util.h"

namespace torch_mlu {
namespace ops {

void cnnl_maximum_internal(
    at::Tensor& output,
    const at::Tensor& input_t,
    const at::Tensor& other_t) {
  if (output.numel() == 0)
    return;
  // Note: Not support pass one CPU tensor that dim is 0 to CNNL kernel.
  auto input = scalar_to_tensor_with_dtype(input_t, output.scalar_type());
  auto other = scalar_to_tensor_with_dtype(other_t, output.scalar_type());
  auto layout = suggest_cnnl_layout(output);

  // get tensor impl
  auto input_impl = getMluTensorImpl(input);
  auto descInput = getTensorDesc(input_impl, layout);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto other_impl = getMluTensorImpl(other);
  auto descOther = getTensorDesc(other_impl, layout);
  auto other_ptr = mlu_data_ptr(other_impl);

  auto output_impl = getMluTensorImpl(output);
  auto descOutput = getTensorDesc(output_impl, layout);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // get workspace size
  size_t tmp_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetMaximumWorkspaceSize(handle, descOutput.get(), &tmp_size));
  // call cnnl max api
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(tmp_size);
  TORCH_CNNL_CHECK(cnnlMaximum(
      handle,
      descInput.get(),
      input_ptr,
      descOther.get(),
      other_ptr,
      descOutput.get(),
      output_ptr,
      workspace_ptr.get(),
      tmp_size));
}

} // namespace ops
} // namespace torch_mlu
