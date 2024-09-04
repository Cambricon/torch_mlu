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

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_native_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    double eps) {
  c10::MaybeOwned<at::Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& weight = *weight_maybe_owned;
  c10::MaybeOwned<at::Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& bias = *bias_maybe_owned;
  const int normalized_ndim = normalized_shape.size();

  at::native::_check_layer_norm_inputs(input, normalized_shape, weight, bias);

  auto axis = input.dim() - normalized_ndim;
  std::vector<int64_t> mean_rstd_size(axis, 1);
  for (int64_t i = 0; i < axis; i++) {
    mean_rstd_size[i] = input.size(i);
  }

  auto input_contiguous = cnnl_contiguous(input, at::MemoryFormat::Contiguous);

  at::Tensor weight_contiguous, bias_contiguous;
  // There are four combinations of the parameters weight and bias:
  // - not None, not None
  // - None, not None
  // - not None, None
  // - None, None
  // As of cnnl 1.26.2, cnnlLayerNormForward only supports cases where both
  // weight and bias are either None or not None.
  // Therefore, for cases not supported by cnnlLayerNormForward, default values
  // need to be manually filled.
  // TODO(miaochen): remove this when PYTORCH-10255 finish
  if (!weight.defined() && !bias.defined()) {
    weight_contiguous = weight;
    bias_contiguous = bias;
  } else if (!weight.defined() && bias.defined()) {
    weight_contiguous = at::ones(normalized_shape, input.options());
    bias_contiguous = cnnl_contiguous(bias, at::MemoryFormat::Contiguous);
  } else if (weight.defined() && !bias.defined()) {
    weight_contiguous = cnnl_contiguous(weight, at::MemoryFormat::Contiguous);
    bias_contiguous = at::zeros(normalized_shape, input.options());
  } else {
    weight_contiguous = cnnl_contiguous(weight, at::MemoryFormat::Contiguous);
    bias_contiguous = cnnl_contiguous(bias, at::MemoryFormat::Contiguous);
  }

  auto output = at::empty(input_contiguous.sizes(), input.options());
  auto acc_type = at::toAccumulateType(input.scalar_type(), /*is_cuda=*/true);
  auto mean = at::empty(mean_rstd_size, input.options().dtype(acc_type));
  auto rstd = at::empty(mean_rstd_size, input.options().dtype(acc_type));

  if (input_contiguous.numel() > 0) {
    cnnl_native_layer_norm_internal(
        input_contiguous,
        weight_contiguous,
        bias_contiguous,
        output,
        mean,
        rstd,
        eps,
        axis);
  }
  const auto input_shape = input.sizes();

  std::vector<int64_t> stat_shape;
  for (size_t idx = 0; idx < axis; ++idx) {
    stat_shape.push_back(input_shape[idx]);
  }
  for (size_t idx = axis; idx < input.dim(); ++idx) {
    stat_shape.push_back(1);
  }

  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);

  return std::make_tuple(std::move(output), std::move(mean), std::move(rstd));
}

} // namespace ops
} // namespace torch_mlu
