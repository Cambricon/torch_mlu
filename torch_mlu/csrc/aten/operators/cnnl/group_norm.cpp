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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

template <typename T>
void check_group_norm_inputs(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    T C,
    int64_t num_groups) {
  TORCH_CHECK(
      num_groups > 0,
      "Expected num groups to be greater than 0, got ",
      num_groups);
  TORCH_CHECK(
      C % num_groups == 0,
      "Expected number of channels in input to be divisible by ",
      "num_groups, but got input of shape ",
      input.sizes(),
      " and "
      "num_groups=",
      num_groups);
  TORCH_CHECK(
      !weight.defined() ||
          (weight.dim() == 1 && at::symint::numel<T>(weight) == C),
      "Expected weight to be a vector of size equal to the number of ",
      "channels in input, but got weight of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
  TORCH_CHECK(
      !bias.defined() || (bias.dim() == 1 && at::symint::numel<T>(bias) == C),
      "Expected bias to be a vector of size equal to the number of ",
      "channels in input, but got bias of shape ",
      weight.sizes(),
      " and input of shape ",
      input.sizes());
}

at::Tensor cnnl_group_norm(
    const at::Tensor& input,
    int64_t num_groups,
    const std::optional<at::Tensor>& weight_opt /* optional */,
    const std::optional<at::Tensor>& bias_opt /* optional */,
    double eps,
    bool /* cudnn_enabled, deprecated */) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });

  const int64_t N = input.size(0);
  const int64_t C = input.size(1);
  check_group_norm_inputs(input, weight, bias, C, num_groups);

  const auto input_shape = input.sizes();
  const int64_t HxW =
      c10::multiply_integers(input_shape.cbegin() + 2, input_shape.cend());

  const at::Tensor kEmpty;
  // Currently, CNNL only support NCHW, NHWC, NDHWC, NLC.
  const auto& X = input.is_contiguous(input.suggest_memory_format())
      ? input
      : input.contiguous();
  const auto& gamma = weight.defined() ? weight.contiguous() : kEmpty;
  const auto& beta = bias.defined() ? bias.contiguous() : kEmpty;
  TORCH_CHECK(!gamma.defined() || gamma.numel() == C);
  TORCH_CHECK(!beta.defined() || beta.numel() == C);
  return std::get<0>(
      at::native_group_norm(X, gamma, beta, N, C, HxW, num_groups, eps));
}

at::Tensor cnnl_group_norm_autograd(
    const at::Tensor& input,
    int64_t num_groups,
    const std::optional<at::Tensor>& weight_opt /* optional */,
    const std::optional<at::Tensor>& bias_opt /* optional */,
    double eps,
    bool cudnn_enabled /* deprecated */) {
  return cnnl_group_norm(
      input, num_groups, weight_opt, bias_opt, eps, cudnn_enabled);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_native_group_norm(
    const at::Tensor& input,
    const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps) {
  const at::Tensor& weight = *at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& bias =
      c10::value_or_else(bias_opt, [] { return at::Tensor(); });

  check_group_norm_inputs(input, weight, bias, C, group);
  TORCH_CHECK(
      input.numel() == N * C * HxW,
      "The size of input: ",
      input.numel(),
      " is not same as GroupNorm size: ",
      N,
      "*",
      C,
      "*",
      HxW);
  TORCH_CHECK(
      input.scalar_type() == at::kFloat ||
          input.scalar_type() == at::kBFloat16 ||
          input.scalar_type() == at::kHalf ||
          input.scalar_type() == at::kDouble,
      "GroupNorm only support float, bfloat16, half and double type inputs, but got dtype: ",
      input.scalar_type());
  TORCH_CHECK(
      (!weight.defined() || input.scalar_type() == weight.scalar_type()) &&
          (!bias.defined() || input.scalar_type() == bias.scalar_type()),
      "GroupNorm only support same dtypes of input, weight and bias, but got ",
      "input dtype: ",
      input.scalar_type(),
      " weight dtype: ",
      weight.scalar_type(),
      " bias dtype: ",
      bias.scalar_type());
  TORCH_CHECK(
      input.is_contiguous(input.suggest_memory_format()),
      "Expected input to be a contiguous or channels_last contiguous tensor.")

  // TODO(sifengyang): cnnlGroupNormForward_v3 will support gamma and beta are
  // None, and remove cnnl_view in CNNLCORE-12619.
  at::Tensor input_internal =
      input.is_contiguous() ? cnnl_view(input, {N, C, 1, HxW}) : input;
  auto mean = at::empty({N, group}, input.options());
  auto rstd = at::empty({N, group}, input.options());
  auto output = at::native::empty_like(input_internal);
  std::tie(output, mean, rstd) = cnnl_group_norm_internal(
      output, input_internal, weight, bias, mean, rstd, eps, group);
  auto out = input.is_contiguous() ? cnnl_view(output, input.sizes()) : output;
  return std::make_tuple(out, mean, rstd);
}

} //  namespace ops
} //  namespace torch_mlu
