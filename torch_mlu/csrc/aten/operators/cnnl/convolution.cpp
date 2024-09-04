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

#include "aten/operators/cnnl/convolution.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "ATen/native/NonSymbolicBC.h"

namespace torch_mlu {
namespace ops {

static const std::set<at::ScalarType> conv_support_dtype{
    at::ScalarType::Half,
    at::ScalarType::Float,
    at::ScalarType::BFloat16,
    at::ScalarType::Double};

at::Tensor cnnl__convolution(
    const at::Tensor& input_r,
    const at::Tensor& weight_r,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    bool /* benchmark */,
    bool /* deterministic */,
    bool /* cudnn_enabled */,
    bool /* allow_tf32 */) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias_r = *bias_maybe_owned;
  auto input = input_r;
  auto weight = weight_r;
  auto bias = bias_r;
  auto k = weight.ndimension();
  c10::IntArrayRef weight_sizes = weight.sizes();
  int64_t dim = k - 2;

  TORCH_MLU_CHECK(dim > 0, "weight should have at least three dimensions");
  TORCH_MLU_CHECK(groups > 0, "non-positive groups is not supported");
  TORCH_MLU_CHECK(
      conv_support_dtype.find(input.scalar_type()) != conv_support_dtype.end(),
      "Convolution mlu op not implemented for '",
      input.scalar_type(),
      "'");
  ConvParams<int64_t> params;
  params.stride = expand_param_if_needed(stride, "stride", dim);
  params.padding = expand_param_if_needed(padding, "padding", dim);
  params.dilation = expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding =
      expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;
  // reserved for future.
  params.benchmark = false;
  params.deterministic = false;
  params.allow_tf32 = torch_mlu::Global::instance().allowCNNLTF32();
  check_shape_forward(input, weight_sizes, bias, params);

  at::Tensor output;
  // don't send empty inputs through backends
  if (input.size(0) == 0 || input.size(1) == 0) {
    auto weight_view = at::native::reshape(weight, -1);
    output = (input.size(1) == 0) ? (input.view(-1) * weight_view)
                                  : (input * weight_view[0]);
    if (bias.defined()) {
      output.add_(bias[0]);
    }
    output = output.view(calc_output_size(input, weight, params));
    return output;
  } else if (input.numel() == 0) {
    TORCH_CHECK(
        false,
        "Only zero batch or zero channel inputs",
        " are supported, but got input shape: ",
        input.sizes());
  }

  // convert conv1d to conv2d.
  if (k == 3) {
    params.view1d_as_2d();
    input = view4d(input);
    weight = view4d(weight);
  }

  check_input_same_type_as_parameters(input, weight, bias);
  if (params.transposed) {
    output = cnnl_convolution_transpose_forward(input, weight, params);
    // add bias
    if (bias.defined()) {
      output.add_(at::native::reshape_bias(input.dim(), bias));
    }
  } else {
    output = cnnl_convolution_normal_forward(
        "cnnl_convolution_normal_forward", input, weight, bias, params);
  }
  if (k == 3) {
    output = view3d(output);
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_convolution_backward(
    const at::Tensor& grad_output_,
    const at::Tensor& input_,
    const at::Tensor& weight_,
    const at::OptionalIntArrayRef bias_sizes_opt,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    bool transposed,
    at::IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  auto grad_output = grad_output_;
  auto input = input_;
  auto weight = weight_;

  auto k = weight.ndimension();
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParams<int64_t> params;
  params.stride = expand_param_if_needed(stride, "stride", dim);
  params.padding = expand_param_if_needed(padding, "padding", dim);
  params.dilation = expand_param_if_needed(dilation, "dilation", dim);
  params.transposed = transposed;
  params.output_padding =
      expand_param_if_needed(output_padding, "output_padding", dim);
  params.groups = groups;
  // reserved for future.
  params.benchmark = false;
  params.deterministic = false;
  params.allow_tf32 = torch_mlu::Global::instance().allowCNNLTF32();

  // Validate inputs.
  check_shape_backward(input, weight.sizes(), params);
  TORCH_CHECK(
      input.dim() == grad_output.dim(),
      "Expected input and grad_output to have the same number of dimensions, but got: ",
      input.dim(),
      " and ",
      grad_output.dim());

  TORCH_MLU_CHECK(
      conv_support_dtype.find(input.scalar_type()) != conv_support_dtype.end(),
      "Convolution mlu op not implemented for '",
      input.scalar_type(),
      "'");

  // output_padding is only supported for transposed convolutions
  if (!params.transposed) {
    for (auto pad : params.output_padding) {
      TORCH_CHECK(
          pad == 0,
          "output_padding is not supported",
          " for non-transposed convolutions; got: ",
          params.output_padding);
    }
  }

  at::Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
  // don't send empty inputs through backends
  if (input.size(0) == 0 || input.size(1) == 0) {
    if (output_mask[0]) {
      backend_grad_input = at::zeros_like(input);
    }
    if (output_mask[1]) {
      backend_grad_weight = at::zeros_like(weight);
    }
    if (output_mask[2]) {
      backend_grad_bias = at::zeros(*bias_sizes_opt, weight.options());
    }
    return std::make_tuple(
        backend_grad_input, backend_grad_weight, backend_grad_bias);
  } else if (input.numel() == 0) {
    TORCH_CHECK(
        false,
        "Only zero batch or zero channel inputs",
        " are supported, but got input shape: ",
        input.sizes());
  }

  // Expand 1d -> 2d.
  // This is only done for backends that don't natively support 1d spatial
  // input.
  if (k == 3) {
    params.view1d_as_2d();
    grad_output = view4d(grad_output);
    input = view4d(input);
    weight = view4d(weight);
  }

  // Only depthwise conv without this check.
  check_input_same_type_as_parameters(input, weight);
  if (params.transposed) {
    std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        cnnl_convolution_transpose_backward(
            input, grad_output, weight, params, output_mask);
  } else {
    std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
        cnnl_convolution_normal_backward(
            input, grad_output, weight, params, output_mask);
  }

  // Convert 2D inputs back to 1D for backends that don't natively support 1D
  // spatial inputs.
  if (output_mask[0] && k == 3) {
    backend_grad_input = view3d(backend_grad_input);
  }
  if (output_mask[1] && k == 3) {
    backend_grad_weight = view3d(backend_grad_weight);
  }
  return std::make_tuple(
      backend_grad_input, backend_grad_weight, backend_grad_bias);
}

} // namespace ops
} // namespace torch_mlu
