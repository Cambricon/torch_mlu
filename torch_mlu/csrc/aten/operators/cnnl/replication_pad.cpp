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

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/cnnl_util.h"
#include "aten/utils/dispatch.h"
#include "aten/utils/tensor_util.h"

namespace torch_mlu {
namespace ops {

TORCH_META_FUNC(replication_pad2d_out_mlu)
(const Tensor& input, IntArrayRef paddingSize) {
  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];
  int64_t pad_t = paddingSize[2];
  int64_t pad_b = paddingSize[3];
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  // allow 0 dim batch size and nothing else.
  bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
  TORCH_CHECK(
      (input.dim() == 3 && input.size(0) != 0 && valid_dims) ||
          (input.dim() == 4 && valid_dims && input.size(3) != 0),
      "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size"
      "and other non-zero dimensions for input, but got: ",
      input.sizes());

  if (input.dim() == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth = iwidth + pad_l + pad_r;

  TORCH_CHECK(
      owidth >= 1 || oheight >= 1,
      "input (H: ",
      iheight,
      ", W: ",
      iwidth,
      " ) is too small."
      " Calculated output H: ",
      oheight,
      " W: ",
      owidth);

  // memory_format needs to be set for following reasons,
  // so meta func is overrided.
  if (input.dim() == 3) {
    // We don't want to create a temporary tensor to infer memory_format here,
    // so we hardcode it as a 3D version of "channels last", hopefully 3D inputs
    // are rare.
    set_output_raw_strided(
        0,
        {nslices, oheight, owidth},
        get_channels_last_strides({nslices, oheight, owidth}),
        input.options());
  } else {
    // here memory_format of output is hardcoded to the same with input, to
    // avoid unnecessary transpose.
    set_output_raw_strided(
        0,
        {nbatch, nslices, oheight, owidth},
        {},
        input.options().memory_format(input.suggest_memory_format()));
  }
}

TORCH_IMPL_FUNC(replication_pad2d_out_mlu)
(const at::Tensor& input, at::IntArrayRef padding, const at::Tensor& output) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "replication_pad2d_out",
      [&] {
        if (input.numel() == 0) {
          return;
        }
        auto input_ = input.dim() == 3 ? at::unsqueeze(input, 0) : input;
        auto output_ = input.dim() == 3 ? at::unsqueeze(output, 0) : output;
        auto input_contiguous =
            cnnl_contiguous(input_, input_.suggest_memory_format());
        auto output_contiguous =
            cnnl_contiguous(output_, input_.suggest_memory_format());
        cnnl_replication_pad2d_internal(
            output_contiguous, input_contiguous, padding);
        if (input.dim() == 3) { // cnnl only support batch mode.
          output_contiguous.squeeze_(0);
        }
        if (is_copy_necessary(output, output_contiguous)) {
          output.copy_(output_contiguous);
        }
      });
}

// TODO(*): currently, replication pad 1D uses cnnlReplicationPad2d, and this
// func takes extra squeeze, unsqueeze and copy calls. Re-adapt this operator
// after cnnlReplicationPad1d is ready.
TORCH_IMPL_FUNC(replication_pad1d_out_mlu)
(const at::Tensor& input, at::IntArrayRef padding, const at::Tensor& output) {
  auto padding_vec = padding.vec();
  std::vector<int64_t> pad(4);
  for (int i = 0; i < padding_vec.size(); i++) {
    pad[i] = static_cast<int>(padding_vec[i]);
  }
  pad[2] = 0;
  pad[3] = 0;
  at::IntArrayRef padding_(pad);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "replication_pad1d_out",
      [&] {
        if (input.numel() == 0) {
          return;
        }
        auto input_ = input.dim() == 2 ? at::unsqueeze(input, 0) : input;
        input_ = at::unsqueeze(input_, 2);
        auto output_ = output.dim() == 2 ? at::unsqueeze(output, 0) : output;
        output_ = at::unsqueeze(output_, 2);
        auto input_contiguous =
            cnnl_contiguous(input_, input_.suggest_memory_format());
        auto output_contiguous =
            cnnl_contiguous(output_, input_.suggest_memory_format());
        cnnl_replication_pad2d_internal(
            output_contiguous, input_contiguous, padding_);
        output_contiguous.squeeze_(2);
        if (input.dim() == 2) { // cnnl only support batch mode.
          output_contiguous.squeeze_(0);
        }
        if (is_copy_necessary(output, output_contiguous)) {
          output.copy_(output_contiguous);
        }
      });
}

} // namespace ops
} // namespace torch_mlu
