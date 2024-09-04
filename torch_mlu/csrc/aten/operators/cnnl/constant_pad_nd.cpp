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

at::Tensor cnnl_constant_pad_nd(
    const at::Tensor& self,
    at::IntArrayRef pad,
    const at::Scalar& value) {
  TORCH_CHECK(
      pad.size() % 2 == 0,
      "Length of pad must be even but instead it equals ",
      pad.size());

  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);

  bool all_pads_is_zero = true;
  for (const auto& i : pad) {
    if (i != 0) {
      all_pads_is_zero = false;
      break;
    }
  }

  // if all of the pads are zero we can optimize and just return the result
  // by copy input.
  // Why not use narrow for negetive pads like Pytorch?
  // Narrow will bring IO overheads because MLU do not support stride and our
  // kernel can support negative pads directly.
  if (all_pads_is_zero) {
    return self_contiguous.clone();
  }

  auto input_sizes = self.sizes();
  auto l_inp = self.dim();
  auto l_pad = pad.size() / 2;
  auto l_diff = l_inp - l_pad;
  TORCH_CHECK(
      l_inp >= (int64_t)l_pad,
      "Length of pad should be no more than twice the number of "
      "dimensions of the input. Pad length is ",
      pad.size(),
      " while the input has ",
      l_inp,
      " dimensions.");

  std::vector<int64_t> new_shape;
  // for MLU pad
  int new_pad[l_inp][2], new_pad_trans[l_inp][2];
  for (size_t i = 0; i < (size_t)l_diff; i++) {
    new_shape.emplace_back(input_sizes[i]);
    new_pad[i][0] = new_pad[i][1] = 0;
  }

  for (size_t i = 0; i < (size_t)l_pad; i++) {
    auto pad_idx = pad.size() - ((i + 1) * 2);
    auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
    TORCH_CHECK(
        new_dim >= 0,
        "The input size ",
        input_sizes[l_diff + i],
        ", plus negative padding ",
        pad[pad_idx],
        " and ",
        pad[pad_idx + 1],
        " resulted in a negative output size, "
        "which is invalid. Check dimension ",
        l_diff + i,
        " of your input.");
    new_shape.emplace_back(new_dim);
    new_pad[l_diff + i][0] = pad[pad_idx];
    new_pad[l_diff + i][1] = pad[pad_idx + 1];
  }

  auto output = at::empty(new_shape, self.options(), memory_format);
  if (memory_format == at::MemoryFormat::ChannelsLast ||
      memory_format == at::MemoryFormat::ChannelsLast3d) {
    new_pad_trans[0][0] = new_pad[0][0];
    new_pad_trans[0][1] = new_pad[0][1];
    for (size_t i = 0; i < (size_t)l_inp - 1; i++) {
      new_pad_trans[i + 1][0] = new_pad[(i + 1) % (l_inp - 1) + 1][0];
      new_pad_trans[i + 1][1] = new_pad[(i + 1) % (l_inp - 1) + 1][1];
    }
    return cnnl_constant_pad_nd_internal(
        output, self_contiguous, new_pad_trans, value, memory_format);
  }

  return cnnl_constant_pad_nd_internal(
      output, self_contiguous, new_pad, value, memory_format);
}

} // namespace ops
} // namespace torch_mlu
