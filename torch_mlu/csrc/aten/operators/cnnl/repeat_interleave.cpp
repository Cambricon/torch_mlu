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

#include "ATen/Dispatch.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/cnnl_util.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {
/*
[NOTE]:
torch.repeat_interleave(self, repeat, output_size) uses `index_select` and this
kernel fucntion to repeat self Tensor. This kernel function computes the
repeated index according to the `repeats`. e.g. input1: repeats = [2, 2]
output_size=None output1 = [0, 0, 1, 1]

input2: repeats = [2, 2, 2], output_size = (2, 3)
result2 = [[0, 0, 1], [1, 2, 2]]
*/
at::Tensor cnnl_repeat_interleave(
    const at::Tensor& repeats,
    c10::optional<int64_t> output_size) {
  TORCH_CHECK(
      repeats.dim() == 1, "repeat_interleave only accept 1D vector as repeat");
  TORCH_CHECK(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      "repeats has to be Long or Int tensor");
  if (repeats.size(0) == 0) {
    return at::empty_like(repeats, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }
  at::Tensor cumsum = repeats.cumsum(0);
  int64_t total;
  if (output_size.has_value()) {
    total = output_size.value();
  } else {
    total = cumsum[-1].item<int64_t>();
    TORCH_CHECK(
        (repeats >= 0).all().item<uint8_t>(), "repeats can not be negative");
  }
  // since cnnlRepeatInterleave repeat input tensor directly, we have no input
  // Tensor in this function, we need to generate input for internal function
  // which is index:[0, 1, 2, ... , self[dim] - 1]. See [NOTE] for more info.
  int steps = repeats.sizes().vec()[0];
  at::Tensor index =
      cast_long_to_int_if_needed(at::empty({steps}, repeats.options()));
  int end = (steps - 1 > 0) ? (steps - 1) : 0;
  cnnl_linspace_internal(index, at::Scalar(0), at::Scalar(end), end);

  at::Tensor output = at::empty({total}, repeats.options());
  auto output_contiguous = cnnl_contiguous(output);
  auto repeats_contiguous = cnnl_contiguous(repeats);
  AT_DISPATCH_INDEX_TYPES(
      repeats.scalar_type(), "cnnl_repeat_interleave", [&]() {
        output_contiguous = create_int_tensor_if_needed(output_contiguous);
        repeats_contiguous = cast_long_to_int_if_needed(repeats_contiguous);
        cnnl_repeat_interleave_internal(
            output_contiguous, index, repeats_contiguous);
        if (is_copy_necessary(output, output_contiguous)) {
          output.copy_(output_contiguous);
        }
      });
  return output;
}
} // namespace ops
} // namespace torch_mlu
