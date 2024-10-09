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

#include <ATen/native/Pool.h>
#include "ATen/core/TensorBody.h"
#include "ATen/core/interned_strings.h"
#include "ATen/ops/empty.h"
#include "ATen/ops/avg_pool1d_native.h"
#include "ATen/ops/max_pool1d_with_indices_native.h"
#include "aten/utils/dispatch.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "c10/core/ScalarType.h"

using at::native::avg_pool2d_backward_shape_check;
using at::native::avg_pool3d_backward_shape_check;
using at::native::max_pool2d_backward_shape_check;
using at::native::max_pool3d_backward_shape_check;
using at::native::pool2d_shape_check;
using at::native::pool3d_shape_check;
using at::native::pooling_output_shape;
using at::native::safe_downcast;
using DimnameList = c10::ArrayRef<at::Dimname>;

namespace torch_mlu {
namespace ops {

#define MAXPOOL2D_KERNEL_MAX 65535

TORCH_PRECOMPUTE_META_FUNC(avg_pool2d_out_mlu)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int64_t kH = kernel_size[0];
  const int64_t kW = kernel_size.size() == 1 ? kH : kernel_size[1];

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int64_t dH = stride.empty() ? kH : stride[0];
  const int64_t dW = stride.empty() ? kW : stride.size() == 1 ? dH : stride[1];

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int64_t padH = padding[0];
  const int64_t padW = padding.size() == 1 ? padH : padding[1];

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  pool2d_shape_check(
      input,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      1,
      1,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output */
  if (input.ndimension() == 3) {
    set_output_raw_strided(
        0,
        {nInputPlane, outputHeight, outputWidth},
        c10::get_channels_last_strides_2d(
            {nInputPlane, outputHeight, outputWidth}),
        input.options());
  } else {
    set_output_raw_strided(
        0,
        {nbatch, nInputPlane, outputHeight, outputWidth},
        {},
        input.options().memory_format(at::MemoryFormat::ChannelsLast));
  }

  return TORCH_PRECOMPUTE_STRUCT(avg_pool2d_out_mlu)()
      .set_kH(kH)
      .set_kW(kW)
      .set_dH(dH)
      .set_dW(dW)
      .set_padH(padH)
      .set_padW(padW);
}

TORCH_META_FUNC(avg_pool2d_backward_out_mlu)
(const Tensor& gradOutput_,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints");
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints");
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "avg_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3); // number of channels (or colors)
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  avg_pool2d_backward_shape_check(
      input,
      gradOutput_,
      nbatch,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output */
  if (input.ndimension() == 3) {
    set_output_raw_strided(
        0,
        input.sizes(),
        get_channels_last_strides_2d(input.sizes()),
        input.options());
  } else {
    set_output_raw_strided(
        0,
        input.sizes(),
        {},
        input.options().memory_format(at::MemoryFormat::ChannelsLast));
  }
}

TORCH_META_FUNC(avg_pool3d_out_mlu)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.size(0);
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t otime =
      pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  pool3d_shape_check(
      input,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      1,
      1,
      1,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth,
      "avg_pool3d()",
      /*check_input_size=*/true);

  /* resize output */
  if (input.ndimension() == 4) {
    set_output_raw_strided(
        0,
        {nslices, otime, oheight, owidth},
        c10::get_channels_last_strides_3d({nslices, otime, oheight, owidth}),
        input.options());
  } else {
    set_output_raw_strided(
        0,
        {nbatch, nslices, otime, oheight, owidth},
        {},
        input.options().memory_format(at::MemoryFormat::ChannelsLast3d));
  }
}

TORCH_META_FUNC(avg_pool3d_backward_out_mlu)
(const Tensor& gradOutput_,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  /* XXX shape check behavior from TH */
  const int64_t otime_for_shape_check =
      pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight_for_shape_check =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_shape_check =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  avg_pool3d_backward_shape_check(
      input,
      gradOutput_,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      itime,
      iheight,
      iwidth,
      otime_for_shape_check,
      oheight_for_shape_check,
      owidth_for_shape_check,
      "avg_pool3d_backward()");

  /* resize output */
  if (input.ndimension() == 4) {
    set_output_raw_strided(
        0,
        input.sizes(),
        get_channels_last_strides_3d(input.sizes()),
        input.options());
  } else {
    set_output_raw_strided(
        0,
        input.sizes(),
        {},
        input.options().memory_format(at::MemoryFormat::ChannelsLast3d));
  }
}

TORCH_META_FUNC(max_pool2d_with_indices_out_mlu)
(const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(
        input.ndimension() == 4,
        "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  const int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  pool2d_shape_check(
      input,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  /* resize output and indices */
  DimnameList maybe_names = input.has_names() ? input.names() : DimnameList{};
  if (input.ndimension() == 3) {
    set_output_raw_strided(
        0,
        {nInputPlane, outputHeight, outputWidth},
        c10::get_channels_last_strides_2d(
            {nInputPlane, outputHeight, outputWidth}),
        input.options(),
        maybe_names);
    /* indices will contain the locations for each output point */
    set_output_raw_strided(
        1,
        {nInputPlane, outputHeight, outputWidth},
        c10::get_channels_last_strides_2d(
            {nInputPlane, outputHeight, outputWidth}),
        input.options().dtype(at::kLong),
        maybe_names);
  } else {
    set_output_raw_strided(
        0,
        {nbatch, nInputPlane, outputHeight, outputWidth},
        {},
        input.options().memory_format(at::MemoryFormat::ChannelsLast),
        maybe_names);
    /* indices will contain the locations for each output point */
    set_output_raw_strided(
        1,
        {nbatch, nInputPlane, outputHeight, outputWidth},
        {},
        input.options()
            .memory_format(at::MemoryFormat::ChannelsLast)
            .dtype(at::kLong),
        maybe_names);
  }
}

TORCH_META_FUNC(max_pool2d_with_indices_backward_out_mlu)
(const Tensor& gradOutput,
 const Tensor& input,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& indices) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      input.dtype() == gradOutput.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `gradOutput` but got dtype ",
      gradOutput.dtype());

  const auto memory_format = input.suggest_memory_format();
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(
        input.ndimension() == 4,
        "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
  } else if (memory_format == at::MemoryFormat::Contiguous) {
    TORCH_CHECK(
        (input.ndimension() == 3 || input.ndimension() == 4),
        "non-empty 3D or 4D (batch mode) tensor expected for input");
  } else {
    TORCH_CHECK(
        false,
        "Unsupport memory format. Supports only ChannelsLast, Contiguous");
  }

  /* sizes */
  const int64_t nInputPlane = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  /* XXX preserve the existing shape check behavior */
  const int64_t outputHeight_for_shape_check = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  const int64_t outputWidth_for_shape_check = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  max_pool2d_backward_shape_check(
      input,
      gradOutput,
      indices,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight_for_shape_check,
      outputWidth_for_shape_check,
      memory_format);

  if (input.ndimension() == 3) {
    set_output_raw_strided(
        0,
        input.sizes(),
        get_channels_last_strides_2d(input.sizes()),
        input.options(),
        input.has_names() ? input.names() : DimnameList{});
  } else {
    set_output_raw_strided(
        0,
        input.sizes(),
        {},
        input.options().memory_format(at::MemoryFormat::ChannelsLast),
        input.has_names() ? input.names() : DimnameList{});
  }
}

TORCH_IMPL_FUNC(avg_pool2d_out_mlu)
(const Tensor& input_,
 int64_t kH_,
 int64_t kW_,
 int64_t dH_,
 int64_t dW_,
 int64_t padH_,
 int64_t padW_,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override,
 const Tensor& output_) {
  // TODO(lipenghui): divisor_override is not supported currently
  TORCH_MLU_CHECK(
      !divisor_override.has_value(), "divisor_override is not supported");

  at::TensorArg output_arg{output_, "output_", 1};
  at::TensorArg input_arg{input_, "input_", 2};

  checkAllSameMLU("avg_pool2d_out_mlu", {output_arg, input_arg});

  const int kH = safe_downcast<int, int64_t>(kH_);
  const int kW = safe_downcast<int, int64_t>(kW_);

  const int dH = safe_downcast<int, int64_t>(dH_);
  const int dW = safe_downcast<int, int64_t>(dW_);

  const int padH = safe_downcast<int, int64_t>(padH_);
  const int padW = safe_downcast<int, int64_t>(padW_);
  const int32_t count = safe_downcast<int32_t, int64_t>(input_.numel());
  if (count == 0) {
    return;
  }

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 3 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor output = input_.dim() == 3 ? at::unsqueeze(output_, 0) : output_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto result_contiguous = maybe_create_out(
      output,
      output.sizes(),
      get_channels_last_strides_2d(output.sizes()),
      output.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_.scalar_type(),
      "avg_pool2d_out_mlu",
      [&] {
        cnnl_pool2d_internal(
            result_contiguous,
            input_contiguous,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            ceil_mode,
            count_include_pad,
            0);

        if (input_.dim() == 3) { // cnnl only support batch mode.
          result_contiguous.squeeze_(0);
        }

        if (is_copy_necessary(output_, result_contiguous)) {
          output_.copy_(result_contiguous);
        }
      });
}

TORCH_IMPL_FUNC(avg_pool2d_backward_out_mlu)
(const Tensor& gradOutput_,
 const Tensor& input_,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override,
 const Tensor& gradInput_) {
  // TODO(lipenghui): divisor_override is not supported currently
  TORCH_MLU_CHECK(
      !divisor_override.has_value(), "divisor_override is not supported");

  at::TensorArg gradInput_arg{gradInput_, "gradInput_", 1};
  at::TensorArg gradOutput_arg{gradOutput_, "gradOutput_", 2};
  at::TensorArg input_arg{input_, "input_", 3};

  checkAllSameMLU(
      "avg_pool2d_backward_out_mlu",
      {gradInput_arg, gradOutput_arg, input_arg});

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int32_t count = safe_downcast<int32_t, int64_t>(input_.numel());
  if (count == 0) {
    return;
  }

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 3 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor gradOutput =
      input_.dim() == 3 ? at::unsqueeze(gradOutput_, 0) : gradOutput_;
  at::Tensor gradInput =
      input_.dim() == 3 ? at::unsqueeze(gradInput_, 0) : gradInput_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto gradOutput_contiguous = cnnl_contiguous(gradOutput, memory_format);
  auto gradInput_contiguous = maybe_create_out(
      gradInput,
      gradInput.sizes(),
      get_channels_last_strides_2d(gradInput.sizes()),
      gradInput.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "avg_pool2d_backward_out_mlu",
      [&] {
        cnnl_pool2d_backward_internal(
            gradInput_contiguous,
            gradOutput_contiguous,
            input_contiguous,
            {},
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            ceil_mode,
            count_include_pad);
        if (input_.dim() == 3) { // cnnl only support batch mode.
          gradInput_contiguous.squeeze_(0);
        }

        if (is_copy_necessary(gradInput_, gradInput_contiguous)) {
          gradInput_.copy_(gradInput_contiguous);
        }
      });
}

TORCH_IMPL_FUNC(avg_pool3d_out_mlu)
(const Tensor& input_,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override,
 const Tensor& output_) {
  // TODO(lipenghui): divisor_override is not supported currently
  TORCH_CHECK(
      !divisor_override.has_value(), "divisor_override is not supported");

  at::TensorArg output_arg{output_, "output_", 1};
  at::TensorArg input_arg{input_, "input_", 2};

  checkAllSameMLU(__func__, {output_arg, input_arg});

  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  const int32_t count = safe_downcast<int32_t, int64_t>(input_.numel());
  if (count == 0) {
    return;
  }

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 4 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor output = input_.dim() == 4 ? at::unsqueeze(output_, 0) : output_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output_contiguous = maybe_create_out(
      output,
      output.sizes(),
      get_channels_last_strides_3d(output.sizes()),
      output.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "avg_pool3d_out_mlu",
      [&] {
        cnnl_pool3d_internal(
            output_contiguous,
            input_contiguous,
            kT,
            kH,
            kW,
            dT,
            dH,
            dW,
            padT,
            padH,
            padW,
            ceil_mode,
            count_include_pad,
            0);
        if (input_.dim() == 4) { // cnnl only support batch mode.
          output_contiguous.squeeze_(0);
        }
        if (is_copy_necessary(output_, output_contiguous)) {
          output_.copy_(output_contiguous);
        }
      });
}

TORCH_IMPL_FUNC(avg_pool3d_backward_out_mlu)
(const Tensor& gradOutput_,
 const Tensor& input_,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 bool ceil_mode,
 bool count_include_pad,
 c10::optional<int64_t> divisor_override,
 const Tensor& gradInput_) {
  // TODO(lipenghui): divisor_override is not supported currently
  TORCH_CHECK(
      !divisor_override.has_value(), "divisor_override is not supported");

  at::TensorArg gradInput_arg{gradInput_, "gradInput_", 1};
  at::TensorArg gradOutput_arg{gradOutput_, "gradOutput_", 2};
  at::TensorArg input_arg{input_, "input_", 3};

  checkAllSameMLU(__func__, {gradInput_arg, gradOutput_arg, input_arg});

  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (gradOutput_.ndimension() == 4 || gradOutput_.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for gradOutput");

  const int32_t count = safe_downcast<int32_t, int64_t>(input_.numel());
  if (count == 0) {
    return;
  }

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 4 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor gradOutput =
      input_.dim() == 4 ? at::unsqueeze(gradOutput_, 0) : gradOutput_;
  at::Tensor gradInput =
      input_.dim() == 4 ? at::unsqueeze(gradInput_, 0) : gradInput_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto gradOutput_contiguous = cnnl_contiguous(gradOutput, memory_format);
  auto gradInput_contiguous = maybe_create_out(
      gradInput,
      gradInput.sizes(),
      get_channels_last_strides_3d(gradInput.sizes()),
      gradInput.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "avg_pool3d_backward_out_mlu",
      [&] {
        cnnl_pool3d_backward_internal(
            gradInput_contiguous,
            gradOutput_contiguous,
            input_contiguous,
            {},
            kT,
            kH,
            kW,
            dT,
            dH,
            dW,
            padT,
            padH,
            padW,
            ceil_mode,
            count_include_pad);
        if (input_.dim() == 4) {
          gradInput_contiguous.squeeze_(0);
        }
        if (is_copy_necessary(gradInput_, gradInput_contiguous)) {
          gradInput_.copy_(gradInput_contiguous);
        }
      });
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_out_mlu)
(const Tensor& input_,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& output_,
 const Tensor& indices_) {
  at::NoNamesGuard guard;

  int kernel_size_prod = 0;
  if (kernel_size.size() == 1) {
    kernel_size_prod = kernel_size[0] * kernel_size[0];
  } else {
    kernel_size_prod = kernel_size[0] * kernel_size[1];
  }
  TORCH_CHECK(
      kernel_size_prod < MAXPOOL2D_KERNEL_MAX,
      "max_pool2d: The kernel size should be smaller than 65535, while this kernel size is ",
      kernel_size_prod);

  at::TensorArg output_arg{output_, "output_", 1};
  at::TensorArg indices_arg{indices_, "indices_", 2};
  at::TensorArg input_arg{input_, "input_", 3};

  checkAllSameMLU(__func__, {output_arg, indices_arg, input_arg});
  if (output_.numel() == 0) {
    return;
  }

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 3 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor output = input_.dim() == 3 ? at::unsqueeze(output_, 0) : output_;
  at::Tensor indices =
      input_.dim() == 3 ? at::unsqueeze(indices_, 0) : indices_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto output_contiguous = maybe_create_out(
      output,
      output.sizes(),
      get_channels_last_strides_2d(output.sizes()),
      output.options());
  auto indices_contiguous = maybe_create_out(
      indices,
      indices.sizes(),
      get_channels_last_strides_2d(indices.sizes()),
      indices.options());
  indices_contiguous = cnnl_contiguous(indices, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_.scalar_type(),
      "max_pool2d_with_indices_out_mlu",
      [&] {
        cnnl_max_pool2d_with_indices_internal(
            output_contiguous,
            indices_contiguous,
            input_contiguous,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            ceil_mode,
            dilationH,
            dilationW);
        if (input_.dim() == 3) { // cnnl only support batch mode.
          output_contiguous.squeeze_(0);
          indices_contiguous.squeeze_(0);
        }

        if (is_copy_necessary(output_, output_contiguous)) {
          output_.copy_(output_contiguous);
        }

        if (is_copy_necessary(indices_, indices_contiguous)) {
          indices_.copy_(indices_contiguous);
        }
      });
}

TORCH_IMPL_FUNC(max_pool2d_with_indices_backward_out_mlu)
(const Tensor& gradOutput_,
 const Tensor& input_,
 IntArrayRef kernel_size,
 IntArrayRef stride,
 IntArrayRef padding,
 IntArrayRef dilation,
 bool ceil_mode,
 const Tensor& indices_,
 const Tensor& gradInput_) {
  at::NoNamesGuard guard;

  int kernel_size_prod = 0;
  if (kernel_size.size() == 1) {
    kernel_size_prod = kernel_size[0] * kernel_size[0];
  } else {
    kernel_size_prod = kernel_size[0] * kernel_size[1];
  }
  TORCH_CHECK(
      kernel_size_prod < MAXPOOL2D_KERNEL_MAX,
      "max_pool2d: The kernel size should be smaller than 65535, while this kernel size is ",
      kernel_size_prod);

  at::TensorArg gradInput_arg{gradInput_, "gradInput_", 1};
  at::TensorArg gradOutput_arg{gradOutput_, "gradOutput_", 2};
  at::TensorArg input_arg{input_, "input_", 3};
  at::TensorArg indices_arg{indices_, "indices_", 4};

  checkAllSameMLU(
      __func__, {gradInput_arg, gradOutput_arg, input_arg, indices_arg});
  if (gradOutput_.numel() == 0) {
    return;
  }

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  // cnnl only support batch mode.
  at::Tensor input = input_.dim() == 3 ? at::unsqueeze(input_, 0) : input_;
  at::Tensor gradOutput =
      input_.dim() == 3 ? at::unsqueeze(gradOutput_, 0) : gradOutput_;
  at::Tensor gradInput =
      input_.dim() == 3 ? at::unsqueeze(gradInput_, 0) : gradInput_;
  at::Tensor indices =
      input_.dim() == 3 ? at::unsqueeze(indices_, 0) : indices_;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto gradOutput_contiguous = cnnl_contiguous(gradOutput, memory_format);
  auto gradInput_contiguous = maybe_create_out(
      gradInput,
      gradInput.sizes(),
      get_channels_last_strides_2d(gradInput.sizes()),
      gradInput.options());
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input_.scalar_type(),
      "max_pool2d_with_indices_out_mlu",
      [&] {
        cnnl_pool2d_backward_internal(
            gradInput_contiguous,
            gradOutput_contiguous,
            input_contiguous,
            indices_contiguous,
            kH,
            kW,
            dH,
            dW,
            padH,
            padW,
            ceil_mode,
            0,
            dilationH,
            dilationW);

        if (input_.dim() == 3) { // cnnl only support batch mode.
          gradInput_contiguous.squeeze_(0);
        }

        if (is_copy_necessary(gradInput_, gradInput_contiguous)) {
          gradInput_.copy_(gradInput_contiguous);
        }
      });
}

at::Tensor create_out_pooling(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  if (strides.empty()) {
    return at::empty(sizes, options);
  } else {
    return at::empty_strided(sizes, strides, options);
  }
}

void resize_out_pooling(
    const at::Tensor& out,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  TORCH_CHECK(
      options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      options.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      options.device() == out.device(),
      "Expected out tensor to have device ",
      options.device(),
      ", but got ",
      out.device(),
      " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  // MLU side don't check the resized status, cause TORCH_MLU always need
  // contiguous output tensor.
  if (!strides.empty()) {
    TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
    // TODO: avoid the redispatch here
    out.as_strided_(sizes, strides);
  } else if (options.memory_format_opt().has_value()) {
    out.unsafeGetTensorImpl()->empty_tensor_restride(
        *options.memory_format_opt());
  }
}

std::tuple<
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t>
max_pool3d_with_indices_pre_compute(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  int kernel_size_prod = 0;
  if (kernel_size.size() == 1) {
    kernel_size_prod = kernel_size[0] * kernel_size[0] * kernel_size[0];
  } else {
    kernel_size_prod = kernel_size[0] * kernel_size[1] * kernel_size[2];
  }
  TORCH_CHECK(
      kernel_size_prod < MAXPOOL2D_KERNEL_MAX,
      "max_pool3d: The kernel size should be smaller than 65535, while this kernel size is ",
      kernel_size_prod);
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
      "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "max_pool3d: padding must be either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 3,
      "max_pool3d: dilation must be either a single int, or a tuple of "
      "three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
      ? dilationT
      : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
      ? dilationT
      : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);
  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;

  const int64_t otime =
      pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);
  const int64_t oheight =
      pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode);
  const int64_t owidth =
      pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);

  pool3d_shape_check(
      input,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      pT,
      pH,
      pW,
      dilationT,
      dilationH,
      dilationW,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth,
      "cnnl_max_pool3d_with_indices_out()");

  return std::make_tuple(
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      pT,
      pH,
      pW,
      dilationT,
      dilationH,
      dilationW,
      nbatch,
      nslices,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_max_pool3d_with_indices_out(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    at::Tensor& output,
    at::Tensor& indices) {
  auto precompute_results = max_pool3d_with_indices_pre_compute(
      input, kernel_size, stride, padding, dilation, ceil_mode);
  int kT, kH, kW, dT, dH, dW, pT, pH, pW, dilationT, dilationH, dilationW;
  int64_t nbatch, nslices, itime, iheight, iwidth, otime, oheight, owidth;
  std::tie(
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      pT,
      pH,
      pW,
      dilationT,
      dilationH,
      dilationW,
      nbatch,
      nslices,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth) = precompute_results;
  resize_out_pooling(
      output,
      {nbatch, nslices, otime, oheight, owidth},
      {},
      input.options().memory_format(at::MemoryFormat::ChannelsLast3d));
  resize_out_pooling(
      indices,
      {nbatch, nslices, otime, oheight, owidth},
      {},
      input.options()
          .memory_format(at::MemoryFormat::ChannelsLast3d)
          .dtype(at::kLong));

  at::TensorArg output_arg{output, "output", 1};
  at::TensorArg indices_arg{indices, "indices", 2};
  at::TensorArg input_arg{input, "input", 3};

  checkAllSameMLU(__func__, {output_arg, indices_arg, input_arg});

  if (input.numel() == 0) {
    if (input.dim() == 4) {
      output.squeeze_(0);
      indices.squeeze_(0);
    }
    return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
  }

  // cnnl only support batch mode, expand to 5 dimemsion.
  at::Tensor input_ = input.dim() == 4 ? input.unsqueeze(0) : input;

  auto input_contiguous =
      cnnl_contiguous(input_, at::MemoryFormat::ChannelsLast3d);
  auto output_contiguous =
      cnnl_contiguous(output, at::MemoryFormat::ChannelsLast3d);
  auto indices_contiguous =
      cnnl_contiguous(indices, at::MemoryFormat::ChannelsLast3d);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "cnnl_max_pool3d_with_indices_out",
      [&] {
        cnnl_max_pool3d_with_indices_internal(
            output_contiguous,
            indices_contiguous,
            input_contiguous,
            kT,
            kH,
            kW,
            dT,
            dH,
            dW,
            pT,
            pH,
            pW,
            ceil_mode,
            dilationT,
            dilationH,
            dilationW);

        if (is_copy_necessary(output, output_contiguous)) {
          output.copy_(output_contiguous);
        }

        if (is_copy_necessary(indices, indices_contiguous)) {
          indices.copy_(indices_contiguous);
        }
      });
  if (input.dim() == 4) {
    output.squeeze_(0);
    indices.squeeze_(0);
  }
  return std::tuple<at::Tensor&, at::Tensor&>(output, indices);
}

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool3d_with_indices(
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode) {
  at::NoNamesGuard guard;

  auto precompute_results = max_pool3d_with_indices_pre_compute(
      input, kernel_size, stride, padding, dilation, ceil_mode);
  int kT, kH, kW, dT, dH, dW, pT, pH, pW, dilationT, dilationH, dilationW;
  int64_t nbatch, nslices, itime, iheight, iwidth, otime, oheight, owidth;
  std::tie(
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      pT,
      pH,
      pW,
      dilationT,
      dilationH,
      dilationW,
      nbatch,
      nslices,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth) = precompute_results;
  at::Tensor output = create_out_pooling(
      {nbatch, nslices, otime, oheight, owidth},
      {},
      input.options().memory_format(at::MemoryFormat::ChannelsLast3d));
  at::Tensor indices = create_out_pooling(
      {nbatch, nslices, otime, oheight, owidth},
      {},
      input.options()
          .memory_format(at::MemoryFormat::ChannelsLast3d)
          .dtype(at::kLong));

  cnnl_max_pool3d_with_indices_out(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output,
      indices);

  guard.reset();
  at::namedinference::propagate_names(output, input);
  at::namedinference::propagate_names(indices, input);

  return std::tuple<at::Tensor, at::Tensor>(output, indices);
}

at::Tensor& cnnl_max_pool3d_with_indices_backward_out(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices,
    at::Tensor& gradInput) {
  at::TensorArg gradInput_arg{gradInput, "gradInput", 1};
  at::TensorArg gradOutput_arg{gradOutput, "gradOutput", 2};
  at::TensorArg input_arg{input, "input", 3};
  at::TensorArg indices_arg{indices, "indices", 4};

  checkAllSameMLU(
      __func__, {gradInput_arg, gradOutput_arg, input_arg, indices_arg});
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "max_pool3d: kernel_size must either be a single int, or a tuple "
      "of three ints")
  int kernel_size_prod = 0;
  if (kernel_size.size() == 1) {
    kernel_size_prod = kernel_size[0] * kernel_size[0] * kernel_size[0];
  } else {
    kernel_size_prod = kernel_size[0] * kernel_size[1] * kernel_size[2];
  }
  TORCH_CHECK(
      kernel_size_prod < MAXPOOL2D_KERNEL_MAX,
      "max_pool3d: The kernel size should be smaller than 65535, while this kernel size is ",
      kernel_size_prod);
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
      "max_pool3d: stride must either be omitted, a single int, or a "
      "tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "max_pool3d: padding must be either be a single int, or a tuple "
      "of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW =
      padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 3,
      "max_pool3d: dilation must be either a single int, or a tuple of "
      "three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1
      ? dilationT
      : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1
      ? dilationT
      : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "cnnl_max_pool3d_with_indices_backward_out(): ",
      "Expected 4D or 5D input tensor, but got ",
      input.sizes());

  TORCH_CHECK(
      (gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
      "cnnl_max_pool3d_with_indices_backward_out(): ",
      "Expected 4D or 5D gradOutput tensor, but got ",
      gradOutput.sizes());

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);

  const int64_t otime = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  max_pool3d_backward_shape_check(
      input,
      gradOutput,
      indices,
      nslices,
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      pT,
      pH,
      pW,
      dilationT,
      dilationH,
      dilationW,
      itime,
      iheight,
      iwidth,
      otime,
      oheight,
      owidth,
      "cnnl_max_pool3d_with_indices_backward_out()");

  resize_out_pooling(
      gradInput,
      {nbatch, nslices, itime, iheight, iwidth},
      {},
      input.options().memory_format(at::MemoryFormat::ChannelsLast3d));

  if (input.numel() == 0) {
    if (input.dim() == 4) {
      gradInput.squeeze_(0);
    }
    return gradInput;
  }

  at::Tensor work_input = input.dim() == 4 ? input.unsqueeze(0) : input;
  at::Tensor work_grad_output =
      input.dim() == 4 ? gradOutput.unsqueeze(0) : gradOutput;
  at::Tensor work_indices = input.dim() == 4 ? indices.unsqueeze(0) : indices;

  auto input_contiguous =
      cnnl_contiguous(work_input, at::MemoryFormat::ChannelsLast3d);
  auto grad_output_contiguous =
      cnnl_contiguous(work_grad_output, at::MemoryFormat::ChannelsLast3d);
  auto indices_contiguous =
      cnnl_contiguous(work_indices, at::MemoryFormat::ChannelsLast3d);
  auto grad_input_contiguous =
      cnnl_contiguous(gradInput, at::MemoryFormat::ChannelsLast3d);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "cnnl_max_pool3d_with_indices_backward_out",
      [&] {
        cnnl_pool3d_backward_internal(
            grad_input_contiguous,
            grad_output_contiguous,
            input_contiguous,
            indices_contiguous,
            kT,
            kH,
            kW,
            dT,
            dH,
            dW,
            pT,
            pH,
            pW,
            ceil_mode,
            false,
            dilationT,
            dilationH,
            dilationW);

        if (is_copy_necessary(gradInput, grad_input_contiguous)) {
          gradInput.copy_(grad_input_contiguous);
        }
      });
  if (input.dim() == 4) {
    gradInput.squeeze_(0);
  }
  return gradInput;
}

at::Tensor cnnl_max_pool3d_with_indices_backward(
    const at::Tensor& gradOutput,
    const at::Tensor& input,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool ceil_mode,
    const at::Tensor& indices) {
  auto input_ = input.dim() == 4 ? at::unsqueeze(input, 0) : input;
  auto gradInput = at::empty(
      input_.sizes(),
      input_.options().memory_format(at::MemoryFormat::ChannelsLast3d));
  cnnl_max_pool3d_with_indices_backward_out(
      gradOutput,
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      indices,
      gradInput);
  return gradInput;
}

} // namespace ops
} // namespace torch_mlu
