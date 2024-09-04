/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
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

#include <ATen/AccumulateType.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/resize.h"

namespace torch_mlu {
namespace ops {

// CNNL only support 2-5d input.
at::Tensor input_shape_helper(
    const at::Tensor& input,
    int64_t num_features,
    int64_t dim) {
  at::Tensor input_t = input;
  if (dim > 5) {
    input_t = input_t.reshape({input_t.size(0), num_features, -1});
  }
  dim = input_t.dim();
  // PT do not have NLC channel last format currently, so we go NHWC
  if (3 == dim) {
    input_t = input_t.unsqueeze(3);
  }
  return input_t;
}

at::Tensor recover_output_shape_helper(
    const at::Tensor& output,
    c10::IntArrayRef sizes,
    int64_t dim) {
  at::Tensor output_t = output;
  if (3 == dim || dim > 5) {
    output_t = output_t.squeeze(3);
  }
  if (dim > 5) {
    output_t = output_t.reshape(sizes);
  }
  return output_t;
}

// TODO(PYTORCH-9290): It is best to register cudnn_batch_norm, but the
// corresponding reverse operator is not what we need, which may cause
// performance regression, so we maintain the original status and register
// native_batch_norm, but the implementation has a risk of misalignment, eg:
// when train is false, save_mean and save_invstd are not calculated.
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> cnnl_native_batch_norm_out(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double epsilon,
    at::Tensor& output,
    at::Tensor& save_mean_t,
    at::Tensor& save_invstd_t) {
  const bool has_running_mean =
      (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var =
      (running_var_opt.has_value() && running_var_opt->defined());
  TORCH_CHECK(
      has_running_mean == has_running_var,
      "The running_mean and running_var must both be passed in and defined or both not.");
  if (!train) {
    TORCH_CHECK(
        has_running_mean,
        "When training is false, running mean and var are needed.");
  }

  const at::Tensor& weight_t = *at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& bias_t = *at::borrow_from_optional_tensor(bias_opt);
  const at::Tensor& running_mean_t =
      *at::borrow_from_optional_tensor(running_mean_opt);
  const at::Tensor& running_var_t =
      *at::borrow_from_optional_tensor(running_var_opt);

  // TODO(PYTORCH-9290): CNNL requires the dtypes of weight, bias, running_mean,
  // running_var, save_mean and save_invstd to be the same, may be optimized
  // after we truly support native_batch_norm.
  at::Tensor weight_acc_t = weight_t;
  at::Tensor bias_acc_t = bias_t;
  at::Tensor running_mean_acc_t = running_mean_t;
  at::Tensor running_var_acc_t = running_var_t;
  if (save_mean_t.scalar_type() == at::kFloat) {
    if (weight_t.defined() &&
        (weight_t.scalar_type() == at::kHalf ||
         weight_t.scalar_type() == at::kBFloat16))
      weight_acc_t = weight_t.to(at::kFloat);
    if (bias_t.defined() &&
        (bias_t.scalar_type() == at::kHalf ||
         bias_t.scalar_type() == at::kBFloat16))
      bias_acc_t = bias_t.to(at::kFloat);
    if (running_mean_t.defined() &&
        (running_mean_t.scalar_type() == at::kHalf ||
         running_mean_t.scalar_type() == at::kBFloat16))
      running_mean_acc_t = running_mean_t.to(at::kFloat);
    if (running_var_t.defined() &&
        (running_var_t.scalar_type() == at::kHalf ||
         running_var_t.scalar_type() == at::kBFloat16))
      running_var_acc_t = running_var_t.to(at::kFloat);
  }

  at::TensorArg input{self, "input", 1}, weight{weight_acc_t, "weight", 2},
      bias{bias_acc_t, "bias", 3},
      running_mean{running_mean_acc_t, "running_mean", 4},
      running_var{running_var_acc_t, "running_var", 5},
      save_mean{save_mean_t, "save_mean", 0},
      save_invstd{save_invstd_t, "save_invstd", 0};
  at::CheckedFrom c = "cnnl_native_batch_norm";
  checkAllSameMLU(c, {input, weight, bias, running_mean, running_var});
  at::checkScalarTypes(
      c, input, {at::kHalf, at::kFloat, at::kDouble, at::kBFloat16});
  TORCH_MLU_CHECK(
      self.scalar_type() == output.scalar_type(),
      "input and output must be the same dtype, but output is ",
      output.scalar_type(),
      ", self is ",
      self.scalar_type());
  if (self.scalar_type() == at::kFloat || self.scalar_type() == at::kDouble) {
    at::checkScalarTypes(c, save_mean, {at::kFloat, at::kDouble});
  } else {
    at::checkScalarTypes(
        c, save_mean, {at::kHalf, at::kFloat, at::kDouble, at::kBFloat16});
  }
  at::checkAllSameType(
      c, {weight, bias, running_mean, running_var, save_mean, save_invstd});
  int64_t num_features = self.size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      at::checkNumel(c, t, num_features);
    }
  }

  auto dim = self.dim();
  auto input_t = input_shape_helper(self, num_features, dim);
  auto memory_format = dim == 2
      ? at::MemoryFormat::Contiguous
      : get_channels_last_memory_format(input_t.dim());
  auto input_conti = cnnl_contiguous(input_t, memory_format);
  at::native::resize_output(output, self.sizes());
  auto output_t = output.sizes().equals(input_t.sizes())
      ? output
      : output.reshape(input_t.sizes());
  auto output_conti = cnnl_contiguous(output_t, memory_format);
  memory_format = at::MemoryFormat::Contiguous;
  auto weight_conti = weight_acc_t.defined()
      ? cnnl_contiguous(weight_acc_t, memory_format)
      : weight_acc_t;
  auto bias_conti = bias_acc_t.defined()
      ? cnnl_contiguous(bias_acc_t, memory_format)
      : bias_acc_t;
  auto running_mean_conti = running_mean_acc_t.defined()
      ? cnnl_contiguous(running_mean_acc_t, memory_format)
      : running_mean_acc_t;
  auto running_var_conti = running_var_acc_t.defined()
      ? cnnl_contiguous(running_var_acc_t, memory_format)
      : running_var_acc_t;
  at::native::resize_output(save_mean_t, {num_features});
  at::native::resize_output(save_invstd_t, {num_features});
  auto save_mean_conti = cnnl_contiguous(save_mean_t, memory_format);
  auto save_invstd_conti = cnnl_contiguous(save_invstd_t, memory_format);

  cnnl_native_batch_norm_internal(
      output_conti,
      input_conti,
      weight_conti,
      bias_conti,
      running_mean_conti,
      running_var_conti,
      save_mean_conti,
      save_invstd_conti,
      train,
      momentum,
      epsilon);

  if (running_mean_t.defined() &&
      running_mean_t.data_ptr() != running_mean_conti.data_ptr()) {
    running_mean_t.copy_(running_mean_conti);
  }
  if (running_var_t.defined() &&
      running_var_t.data_ptr() != running_var_conti.data_ptr()) {
    running_var_t.copy_(running_var_conti);
  }

  auto output_conti_t =
      recover_output_shape_helper(output_conti, self.sizes(), dim);
  if (output.data_ptr() != output_conti_t.data_ptr()) {
    output.copy_(output_conti_t);
  }
  if (save_mean_t.data_ptr() != save_mean_conti.data_ptr()) {
    save_mean_t.copy_(save_mean_conti);
  }
  if (save_invstd_t.data_ptr() != save_invstd_conti.data_ptr()) {
    save_invstd_t.copy_(save_invstd_conti);
  }
  return std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(
      output, save_mean_t, save_invstd_t);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_native_batch_norm(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    bool train,
    double momentum,
    double epsilon) {
  int64_t num_features = self.size(1);
  auto dim = self.dim();
  auto input = input_shape_helper(self, num_features, dim);
  auto memory_format = dim == 2 ? at::MemoryFormat::Contiguous
                                : get_channels_last_memory_format(input.dim());
  auto output = at::empty_like(input, memory_format);
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_mlu=*/true));
  auto save_mean = at::empty({num_features}, options);
  auto save_invstd = at::empty({num_features}, options);

  cnnl_native_batch_norm_out(
      input,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      train,
      momentum,
      epsilon,
      output,
      save_mean,
      save_invstd);

  auto output_t = recover_output_shape_helper(output, self.sizes(), dim);

  return std::make_tuple(output_t, save_mean, save_invstd);
}

void check_dims_match_num_input_features(
    const char* arg_name,
    c10::SymInt expected,
    c10::SymInt actual) {
  TORCH_CHECK(
      actual == expected,
      arg_name,
      " should contain ",
      expected,
      " elements not ",
      actual);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_native_batch_norm_backward(
    const at::Tensor& grad_out_t,
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    const c10::optional<at::Tensor>& save_mean_opt,
    const c10::optional<at::Tensor>& save_invstd_opt,
    bool train,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  const Tensor& weight_t = *at::borrow_from_optional_tensor(weight_opt);
  const Tensor& running_mean_t =
      *at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_var_t =
      *at::borrow_from_optional_tensor(running_var_opt);
  const Tensor& save_mean_t = *at::borrow_from_optional_tensor(save_mean_opt);
  const Tensor& save_invstd_t =
      *at::borrow_from_optional_tensor(save_invstd_opt);

  // TODO(PYTORCH-9290): CNNL requires the dtypes of weight, bias, running_mean,
  // running_var, save_mean and save_invstd to be the same, may be optimized
  // after we truly support native_batch_norm.
  at::Tensor weight_acc_t = weight_t;
  at::Tensor running_mean_acc_t = running_mean_t;
  at::Tensor running_var_acc_t = running_var_t;
  if (save_mean_t.scalar_type() == at::kFloat) {
    if (weight_t.defined() &&
        (weight_t.scalar_type() == at::kHalf ||
         weight_t.scalar_type() == at::kBFloat16))
      weight_acc_t = weight_t.to(at::kFloat);
    if (running_mean_t.defined() &&
        (running_mean_t.scalar_type() == at::kHalf ||
         running_mean_t.scalar_type() == at::kBFloat16))
      running_mean_acc_t = running_mean_t.to(at::kFloat);
    if (running_var_t.defined() &&
        (running_var_t.scalar_type() == at::kHalf ||
         running_var_t.scalar_type() == at::kBFloat16))
      running_var_acc_t = running_var_t.to(at::kFloat);
  }

  at::TensorArg grad_out{grad_out_t, "grad_out", 1}, input{self, "weight", 2},
      weight{weight_acc_t, "weight", 3},
      running_mean{running_mean_acc_t, "running_mean", 4},
      running_var{running_var_acc_t, "running_var", 5},
      save_mean{save_mean_t, "save_mean", 6},
      save_invstd{save_invstd_t, "save_invstd", 7};
  at::CheckedFrom c = "cnnl_native_batch_norm_backward";
  at::checkAllDefined(c, {grad_out, input, save_mean, save_invstd});
  checkAllSameMLU(
      c,
      {grad_out,
       input,
       weight,
       running_mean,
       running_var,
       save_mean,
       save_invstd});
  TORCH_MLU_CHECK(
      grad_out_t.scalar_type() == self.scalar_type(),
      "input and grad_out must be the same dtype, but grad_out is ",
      grad_out_t.scalar_type(),
      ", input is ",
      self.scalar_type());
  at::checkAllSameType(
      c, {weight, running_mean, running_var, save_mean, save_invstd});
  at::checkSameSize(c, input, grad_out);

  int64_t num_features = self.size(1);
  auto dim = self.dim();
  auto input_t = input_shape_helper(self, num_features, dim);
  auto memory_format = dim == 2
      ? at::MemoryFormat::Contiguous
      : get_channels_last_memory_format(input_t.dim());
  auto input_conti = cnnl_contiguous(input_t, memory_format);
  auto grad_out_tt = grad_out_t.sizes().equals(input_t.sizes())
      ? grad_out_t
      : grad_out_t.reshape(input_t.sizes());
  auto grad_out_conti = cnnl_contiguous(grad_out_tt, memory_format);
  memory_format = at::MemoryFormat::Contiguous;
  auto weight_conti = weight_acc_t.defined()
      ? cnnl_contiguous(weight_acc_t, memory_format)
      : weight_acc_t;
  auto running_mean_conti = running_mean_acc_t.defined()
      ? cnnl_contiguous(running_mean_acc_t, memory_format)
      : running_mean_acc_t;
  auto running_var_conti = running_var_acc_t.defined()
      ? cnnl_contiguous(running_var_acc_t, memory_format)
      : running_var_acc_t;
  auto save_mean_conti = cnnl_contiguous(save_mean_t, memory_format);
  auto save_invstd_conti = cnnl_contiguous(save_invstd_t, memory_format);

  // CNNL currently do not support grad_input_mask parameter, please ref
  // PYTORCH-9290.
  auto output = cnnl_native_batch_norm_backward_internal(
      grad_out_conti,
      input_conti,
      weight_conti,
      running_mean_conti,
      running_var_conti,
      save_mean_conti,
      save_invstd_conti,
      train,
      epsilon);
  at::Tensor grad_input = grad_input_mask[0]
      ? recover_output_shape_helper(std::get<0>(output), self.sizes(), dim)
      : at::Tensor();
  at::Tensor grad_weight =
      grad_input_mask[1] ? std::get<1>(output) : at::Tensor();
  at::Tensor grad_bias =
      grad_input_mask[2] ? std::get<2>(output) : at::Tensor();

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace ops
} // namespace torch_mlu
