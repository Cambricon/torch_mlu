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

// Note: for repeat op, output tensor dim may be greater than input tensor dim,
// cnnl op need input tensor dim equal to output tensor dim,
// so normal insert 1 in the begin of input shape.
at::Tensor& cnnl_repeat_internal(at::Tensor& output, const at::Tensor& input) {
  // when input is channels last tensor, reshape will change
  // tensor memory format. Now based on input_size and memory_format
  // generate a new stride.
  auto memory_format = input.suggest_memory_format();
  // TODO(mengpenghui):The historical reason for this check is not clear,
  // temporarily reserved.
  TORCH_MLU_CHECK(
      memory_format != c10::MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");
  const int64_t input_dim = input.dim();
  const int64_t output_dim = output.dim();

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;

  int64_t num_new_dimensions = output_dim - input_dim;
  if (num_new_dimensions > 0) {
    std::vector<int64_t> input_size = input.sizes().vec();
    std::vector<int64_t> input_stride = input.strides().vec();
    int64_t outermost_stride =
        input_dim == 0 ? 1 : input.stride(0) * input.size(0);
    for (int64_t i = 0; i < num_new_dimensions; ++i) {
      input_size.insert(input_size.begin(), 1);
      input_stride.insert(input_stride.begin(), outermost_stride);
    }
    input_desc.set(input, input_size, input_stride, CNNL_LAYOUT_ARRAY);
    output_desc.set(
        output,
        output.sizes().vec(),
        output.strides().vec(),
        CNNL_LAYOUT_ARRAY);
  } else {
    input_desc.set(input);
    output_desc.set(output);
  }

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  TORCH_CNNL_CHECK(cnnlTile(
      handle, input_desc.desc(), input_ptr, output_desc.desc(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
