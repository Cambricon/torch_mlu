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

#pragma once
#include <vector>
#include "c10/util/ArrayRef.h"
#include "ATen/native/ConvUtils.h"

namespace torch_mlu {
namespace ops {

// Copy from pytorch: aten/src/ATen/native/Convolution.cpp
template <typename T>
struct ConvParams {
  std::vector<int64_t> stride;
  std::vector<T> padding;
  std::vector<int64_t> dilation;
  bool transposed;
  std::vector<T> output_padding;
  int groups;
  bool benchmark;
  bool deterministic;
  bool cudnn_enabled;
  bool allow_tf32;

  bool is_strided() const {
    bool is_strided = false;
    for (auto s : stride) {
      is_strided |= (s != 1);
    }
    return is_strided;
  }

  bool is_dilated() const {
    bool is_dilated = false;
    for (auto d : dilation) {
      is_dilated |= (d != 1);
    }
    return is_dilated;
  }

  bool is_padded() const {
    bool is_padded = false;
    for (auto p : padding) {
      is_padded |= (p != 0);
    }
    return is_padded;
  }

  bool is_output_padding_neg() const {
    bool is_non_neg = false;
    for (const auto& p : output_padding) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }

  bool is_output_padding_big() const {
    bool is_big = false;
    for (auto i : c10::irange(output_padding.size())) {
      is_big |= (output_padding[i] >= stride[i]);
    }
    return is_big;
  }

  bool is_padding_neg() const {
    bool is_non_neg = false;
    for (const auto& p : padding) {
      is_non_neg |= (p < 0);
    }
    return is_non_neg;
  }

  bool is_stride_nonpos() const {
    bool is_nonpos = false;
    for (auto s : stride) {
      is_nonpos |= (s <= 0);
    }
    return is_nonpos;
  }

  void view1d_as_2d() {
    if (stride.size() == 1) {
      stride.insert(stride.begin(), 1);
      padding.insert(padding.begin(), 0);
      dilation.insert(dilation.begin(), 1);
      output_padding.insert(output_padding.begin(), 0);
    }
  }

  bool needs_64bit_indexing_no_split(
      const at::Tensor& input,
      const at::Tensor& weight) const {
    constexpr int64_t int_max = std::numeric_limits<int>::max();
    auto numel_input = at::symint::numel<T>(input);
    // empty input
    if (numel_input == 0) {
      return false;
    }
    // input size can not be reduced to the range of int by splitting the batch
    // dim
    auto n = at::symint::size<T>(input, 0);
    if (numel_input / n > int_max) {
      return true;
    }
    // output size can not be reduced to the range of int by splitting the batch
    // dim
    T outsize = 1;
    if (transposed) {
      auto o = conv_input_size(
          at::symint::sizes<T>(input),
          at::symint::sizes<T>(weight),
          padding,
          output_padding,
          stride,
          dilation,
          groups);
      outsize = c10::multiply_integers(o.begin() + 1, o.end());
    } else {
      auto o = conv_output_size(
          at::symint::sizes<T>(input),
          at::symint::sizes<T>(weight),
          padding,
          stride,
          dilation);
      outsize = c10::multiply_integers(o.begin() + 1, o.end());
    }
    return outsize > int_max;
  }
};

// Copy from pytorch: aten/src/ATen/native/Convolution.cpp
template <typename T>
static void check_shape_forward(
    const at::Tensor& input,
    const c10::ArrayRef<T>& weight_sizes,
    const at::Tensor& bias,
    const ConvParams<T>& params) {
  int64_t k = input.ndimension();
  int64_t weight_dim = weight_sizes.size();
  int64_t groups = params.groups;
  const auto& padding = params.padding;
  const auto& dilation = params.dilation;
  bool transposed = params.transposed;

  TORCH_CHECK(!params.is_padding_neg(), "negative padding is not supported");
  TORCH_CHECK(
      !params.is_output_padding_neg(),
      "negative output_padding is not supported");
  TORCH_CHECK(
      !params.is_stride_nonpos(), "non-positive stride is not supported");

  TORCH_CHECK(
      weight_dim == k,
      "Expected ",
      weight_dim,
      "-dimensional input for ",
      weight_dim,
      "-dimensional weight ",
      weight_sizes,
      ", but got ",
      k,
      "-dimensional input of size ",
      at::symint::sizes<T>(input),
      " instead");
  TORCH_CHECK(
      weight_sizes[0] >= groups,
      "Given groups=",
      groups,
      ", expected weight to be at least ",
      groups,
      " at dimension 0, but got weight of size ",
      weight_sizes,
      " instead");
  TORCH_CHECK(
      weight_sizes[0] % groups == 0,
      "Given groups=",
      groups,
      ", expected weight to be divisible by ",
      groups,
      " at dimension 0, but got weight of size [",
      weight_sizes,
      "] instead");

  if (!transposed) {
    std::vector<T> input_shape;
    std::vector<T> kernel_shape;
    bool kernel_size_correct = true;

    TORCH_CHECK(
        at::symint::size<T>(input, 1) == (weight_sizes[1] * groups),
        "Given groups=",
        groups,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        at::symint::sizes<T>(input),
        " to have ",
        (weight_sizes[1] * groups),
        " channels, but got ",
        at::symint::size<T>(input, 1),
        " channels instead");

    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 &&
             at::symint::size<T>(bias, 0) == weight_sizes[0]),
        "Given weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[0],
        " elements",
        ", but got bias of size ",
        at::symint::sizes<T>(bias),
        " instead");

    for (const auto i : c10::irange(2, k)) {
      input_shape.push_back(at::symint::size<T>(input, i) + 2 * padding[i - 2]);
      // log new kernel size considering dilation
      kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
      if (input_shape.back() < kernel_shape.back()) {
        kernel_size_correct = false;
      }
    }

    TORCH_CHECK(
        input_shape.size() == kernel_shape.size(),
        "Inconsistent shape between Input and Kernel");

    if (!kernel_size_correct) {
      // If kernel size is incorrect
      std::ostringstream input_ss;
      std::ostringstream kernel_ss;
      std::string separator = "";

      for (int i = 0, len = input_shape.size(); i < len; ++i) {
        input_ss << separator << input_shape[i];
        kernel_ss << separator << kernel_shape[i];
        separator = " x ";
      }

      AT_ERROR(
          "Calculated padded input size per channel: (",
          input_ss.str(),
          "). "
          "Kernel size: (",
          kernel_ss.str(),
          "). Kernel size can't be greater than actual input size");
    }
  } else { // transposed
    TORCH_CHECK(
        at::symint::size<T>(input, 1) == weight_sizes[0],
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected input",
        at::symint::sizes<T>(input),
        " to have ",
        weight_sizes[0],
        " channels, but got ",
        at::symint::size<T>(input, 1),
        " channels instead");
    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 &&
             at::symint::size<T>(bias, 0) == weight_sizes[1] * groups),
        "Given transposed=",
        transposed,
        ", weight of size ",
        weight_sizes,
        ", expected bias to be 1-dimensional with ",
        weight_sizes[1] * groups,
        " elements",
        ", but got bias of size ",
        at::symint::sizes<T>(bias),
        " instead");
  }
}

template <typename T>
static void check_shape_backward(
    const at::Tensor& input,
    const c10::ArrayRef<T>& weight_sizes,
    const ConvParams<T>& params) {
  check_shape_forward<T>(input, weight_sizes, /*bias=*/at::Tensor(), params);
}

// Copy form aten/src/ATen/native/utils/ParamUtils.h
template <typename T>
inline std::vector<T> _expand_param_if_needed(
    at::ArrayRef<T> list_param,
    const char* param_name,
    int64_t expected_dim) {
  if (list_param.size() == 1) {
    return std::vector<T>(expected_dim, list_param[0]);
  } else if ((int64_t)list_param.size() != expected_dim) {
    std::ostringstream ss;
    ss << "expected " << param_name << " to be a single integer value or a "
       << "list of " << expected_dim << " values to match the convolution "
       << "dimensions, but got " << param_name << "=" << list_param;
    AT_ERROR(ss.str());
  } else {
    return list_param.vec();
  }
}

inline std::vector<int64_t> expand_param_if_needed(
    at::IntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  return _expand_param_if_needed(list_param, param_name, expected_dim);
}

inline std::vector<c10::SymInt> expand_param_if_needed(
    at::SymIntArrayRef list_param,
    const char* param_name,
    int64_t expected_dim) {
  return _expand_param_if_needed(list_param, param_name, expected_dim);
}

// Copy form aten/src/ATen/native/Convolution.cpp
static auto view4d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(
      tensor.ndimension() == 3,
      "expected 3D tensor, got tensor with ",
      tensor.ndimension(),
      " dimensions instead");
  return tensor.unsqueeze(2);
}

// Copy form aten/src/ATen/native/Convolution.cpp
static auto view3d(const at::Tensor& tensor) -> at::Tensor {
  TORCH_CHECK(
      tensor.ndimension() == 4,
      "expected 4D tensor, got tensor with ",
      tensor.ndimension(),
      " dimensions instead");
  return tensor.squeeze(2);
}

// Copy from aten/src/ATen/native/Convolution.cpp
static void check_input_same_type_as_parameters(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  TORCH_CHECK(
      input.options().type_equal(weight.options()),
      "Input type (",
      input.toString(),
      ") and weight type (",
      weight.toString(),
      ") should be the same");
  TORCH_CHECK(
      !bias.defined() || (input.options().type_equal(bias.options())),
      "Input type (",
      input.toString(),
      ") and bias type (",
      bias.toString(),
      ") should be the same");
}

static void check_input_same_type_as_parameters(
    const at::Tensor& input,
    const at::Tensor& weight) {
  check_input_same_type_as_parameters(input, weight, /*bias=*/at::Tensor());
}

// Copy from aten/src/ATen/native/Convolution.cpp
static inline std::vector<int64_t> calc_output_size(
    const at::Tensor& input,
    const at::Tensor& weight,
    const ConvParams<int64_t>& params) {
  std::vector<int64_t> output_size = params.transposed
      ? at::native::conv_input_size(
            input.sizes(),
            weight.sizes(),
            params.padding,
            params.output_padding,
            params.stride,
            params.dilation,
            params.groups)
      : at::native::conv_output_size(
            input.sizes(),
            weight.sizes(),
            params.padding,
            params.stride,
            params.dilation);

  // Handle empty # of channels.
  if (input.size(1) == 0) {
    output_size[1] = 0;
  }
  return output_size;
}

// Almost same with is_depthwise from aten/src/ATen/native/Convolution.cpp
// There are a little different with pytorch gpu side, TORCH_MLU support
// depth-wise for 2-d convolution.
static inline bool is_mlu_depth_wise_conv(
    const at::Tensor& input,
    const at::Tensor& weight,
    int64_t groups) {
  // no point if there is only a single group
  // output channels must be a multiple of input channels
  return input.ndimension() == 4 && input.size(1) == groups &&
      weight.size(1) == 1 && groups > 1 && weight.size(0) % input.size(1) == 0;
}

TORCH_API inline at::MemoryFormat _determine_backend_memory_format(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::native::ConvBackend backend) {
  auto k = weight.ndimension();
  return k == 5 ? at::MemoryFormat::ChannelsLast3d
                : at::MemoryFormat::ChannelsLast;
}

} // namespace ops
} // namespace torch_mlu
