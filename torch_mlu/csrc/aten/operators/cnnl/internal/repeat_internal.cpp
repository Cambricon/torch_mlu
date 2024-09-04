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
  const int64_t input_dim = input.dim();
  const int64_t output_dim = output.dim();

  // get current handle
  auto handle = getCurrentHandle();

  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  tensorDescPtr_t input_desc;
  tensorDescPtr_t output_desc;

  int64_t num_new_dimensions = output_dim - input_dim;
  if (num_new_dimensions > 0) {
    c10::SmallVector<int64_t, 8> input_size(input.sizes());
    c10::SmallVector<int64_t, 8> input_stride(input.strides());
    int64_t outermost_stride =
        input_dim == 0 ? 1 : input.stride(0) * input.size(0);
    for (int64_t i = 0; i < num_new_dimensions; ++i) {
      input_size.insert(input_size.begin(), 1);
      input_stride.insert(input_stride.begin(), outermost_stride);
    }

    cnnlDataType_t input_data_type = getCnnlType(input_impl);
    cnnlDataType_t output_data_type = getCnnlType(output_impl);
    input_desc = getTensorDesc(
        input_size, input_stride, input_data_type, CNNL_LAYOUT_ARRAY);
    output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  } else {
    input_desc = getTensorDesc(input_impl);
    output_desc = getTensorDesc(output_impl);
  }

  TORCH_CNNL_CHECK(cnnlTile(
      handle, input_desc.get(), input_ptr, output_desc.get(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
