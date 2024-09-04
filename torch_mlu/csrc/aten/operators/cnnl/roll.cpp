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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_roll(
    const at::Tensor& self,
    at::IntArrayRef shifts,
    at::IntArrayRef dims) {
  auto input_contiguous = cnnl_contiguous(self, self.suggest_memory_format());
  auto output = at::empty_like(input_contiguous);
  if (output.numel() == 0) {
    return output;
  }

  if (dims.size() != 1 || shifts.size() != 1) {
    TORCH_CHECK(shifts.size() > 0, "`shifts` required");
    if (dims.size() == 0 && shifts.size() == 1) {
      auto flattened = input_contiguous.view(self.numel());
      auto flattened_output = output.view(self.numel());
      return cnnl_roll_internal(flattened_output, flattened, shifts[0], 0)
          .view(self.sizes());
    }
    TORCH_CHECK(
        shifts.size() == dims.size(),
        "shifts and dimensions must align. shifts: ",
        shifts.size(),
        ", dims:",
        dims.size());
    AT_ASSERT(dims.size() > 1);
  }

  return cnnl_roll_internal(output, input_contiguous, shifts, dims);
}

} // namespace ops
} // namespace torch_mlu
