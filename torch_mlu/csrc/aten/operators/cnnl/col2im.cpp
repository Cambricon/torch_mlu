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

#include "ATen/div_rtn.h"
#include "ATen/native/im2col_shape_check.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/cnnl_util.h"
#include "aten/utils/dispatch.h"
#include "aten/utils/internal_util.h"
#include "aten/utils/tensor_util.h"

namespace torch_mlu {
namespace ops {

void col2im_out_mlu_template(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef output_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  at::TensorArg input_arg{input, "input", 1};
  at::TensorArg output_arg{output, "output", 2};
  checkAllSameMLU("col2im_out_mlu", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  at::native::col2im_shape_check(
      input,
      at::Tensor(),
      output_height,
      output_width,
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  bool batched_input = true;

  if (input.dim() == 2) {
    // Force batch
    batched_input = false;
    input.unsqueeze_(0);
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);

  output.resize_({batch_size, n_output_plane, output_height, output_width});

  // CNNL kernel doesn't follow 0 element rules, so torch_mlu skip 0-element in
  // here.
  if (input.numel() != 0) {
    // CNNL kernel side only support NLC, so convert C to last.
    auto input_permute = input.permute(at::IntArrayRef({0, 2, 1}));
    at::Tensor input_contiguous = cnnl_contiguous(input_permute);
    at::Tensor output_contiguous = cnnl_contiguous(output);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "col2im",
        [&] {
          col2im_out_internal(
              output_contiguous,
              input_contiguous,
              kernel_size.vec(),
              dilation.vec(),
              padding.vec(),
              stride.vec());
        });

    if (is_copy_necessary(output, output_contiguous)) {
      output.copy_(output_contiguous);
    }
  }

  if (!batched_input) {
    output.squeeze_(0);
  }
}

at::Tensor& cnnl_col2im_out(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::Tensor& output) {
  col2im_out_mlu_template(
      output, input, output_size, kernel_size, dilation, padding, stride);
  return output;
}

at::Tensor cnnl_col2im(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    at::IntArrayRef kernel_size,
    at::IntArrayRef dilation,
    at::IntArrayRef padding,
    at::IntArrayRef stride) {
  at::Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  cnnl_col2im_out(
      input, output_size, kernel_size, dilation, padding, stride, output);
  return output;
}

} // namespace ops
} // namespace torch_mlu
