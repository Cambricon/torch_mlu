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

#include <algorithm>
#include <stdint.h> // NOLINT
#include "ATen/native/Resize.h" // NOLINT
#include "ATen/NativeFunctions.h" // NOLINT
#include "aten/operators/cnnl/cnnl_kernel.h" // NOLINT
#include "aten/operators/cnnl/internal/cnnl_internal.h" // NOLINT
#include "aten/utils/cnnl_util.h" // NOLINT
#include "aten/utils/internal_util.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/utils.h"

namespace torch_mlu {
// only support original tensor memory format, memory_format just affect output.
// size in shared_storage tensor is always relayable.
at::Tensor cnnl_contiguous(
    const at::Tensor& input,
    c10::MemoryFormat memory_format) {
  if (!input.defined())
    return input;
  // Check tensor device type, and call native contiguous() if cpu tensor.
  if (input.device().is_privateuseone() == false) {
    return input.contiguous(memory_format);
  }
  TORCH_CHECK(
      memory_format != c10::MemoryFormat::Preserve,
      "Preserve memory format is unsupported by the contiguous operator.");
  // Channels last or channels last3d only support 4 or 5 dimensions.
  TORCH_CHECK(
      !(memory_format == at::MemoryFormat::ChannelsLast && input.dim() != 4),
      "required rank 4 tensor to use channels_last format");
  TORCH_CHECK(
      !(memory_format == at::MemoryFormat::ChannelsLast3d && input.dim() != 5),
      "required rank 5 tensor to use ChannelsLast3d format");
  if (input.is_contiguous(memory_format)) {
    return input;
  }
  auto output = at::empty(input.sizes(), input.options(), memory_format);
  torch_mlu::ops::cnnl_copy_internal(output, input);
  return output;
}

} // namespace torch_mlu
