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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {

void cnnl_cumsum_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim) {
  auto input_impl = getMluTensorImpl(input);
  auto descInput = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto descOutput = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();
  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetCumsumWorkspaceSize(handle, descInput.get(), dim, &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  AT_DISPATCH_MLU_INT_AND_FLOAT_TYPES(input.scalar_type(), "cumsum", [&] {
    TORCH_CNNL_CHECK(cnnlCumsum_v2(
        handle,
        descInput.get(),
        input_ptr,
        dim,
        false,
        false, // exclusive and reverse not supported
        CNNL_NOT_PROPAGATE_NAN,
        descOutput.get(),
        output_ptr,
        workspace_ptr.get(),
        space_size));
  });
}

} // namespace ops
} // namespace torch_mlu
