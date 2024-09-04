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

namespace torch_mlu {
namespace ops {

void cnnl_replication_pad2d_internal(
    const at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef padding) {
  TORCH_CHECK(
      self.dim() == 4,
      "currently, 4D (batch mode) tensor expected for input, but got: ",
      self.dim());
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);
  auto padding_vec = padding.vec();
  int pad[4];
  for (int i = 0; i < padding_vec.size(); i++) {
    pad[i] = static_cast<int>(padding_vec[i]);
  }
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  auto suggest_self_layout = suggest_cnnl_layout(self);
  auto suggest_out_layout = suggest_cnnl_layout(output);
  descInput.set(self, suggest_self_layout);
  descOutput.set(output, suggest_out_layout);
  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  TORCH_CNNL_CHECK(cnnlReplicationPad2d(
      handle, descInput.desc(), input_ptr, pad, descOutput.desc(), output_ptr));
}
} // namespace ops
} // namespace torch_mlu
