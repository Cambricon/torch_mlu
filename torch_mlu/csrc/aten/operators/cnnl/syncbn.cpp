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
#include "aten/operators/cnnl/resize.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor> cnnl_batch_norm_stats(
    const at::Tensor& self,
    double epsilon) {
  TORCH_CHECK(self.numel() > 0, "currently do not support empty input as GPU");
  TORCH_CHECK(self.dim() >= 2, "input.dim() must be not less than 2");

  auto epsilon_f = c10::checked_convert<float, double>(epsilon, "float");

  at::Tensor self_reshaped = self;
  // CNNL does not support input larger than 5D
  if (self.dim() > 5) {
    self_reshaped = self.reshape({self.size(0), self.size(1), -1});
  }
  // Pytorch does not support 3D channel last currently
  if (3 == self_reshaped.dim()) {
    self_reshaped = self_reshaped.unsqueeze(3);
  }
  int64_t dim = self_reshaped.dim();
  auto memory_format = dim > 3 ? get_channels_last_memory_format(dim)
                               : LEGACY_CONTIGUOUS_MEMORY_FORMAT;
  auto self_channels_last = cnnl_contiguous(self_reshaped, memory_format);

  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_stats_mlu",
      [&] {
        return cnnl_batch_norm_stats_internal(self_channels_last, epsilon_f);
      });
}

// accepting input(self) here to determine template data types,
// since running_mean/running_var are optional
std::tuple<Tensor, Tensor> cnnl_batch_norm_gather_stats(
    const at::Tensor& self,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    double momentum,
    double epsilon,
    int64_t count) {
  const at::Tensor& running_mean =
      *at::borrow_from_optional_tensor(running_mean_opt);
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  std::vector<int64_t> counts(mean.size(0), count);
  at::Tensor counts_tensor = at::from_blob(
      (void*)counts.data(),
      {(int64_t)counts.size()},
      self.options().dtype(at::kLong).device(at::kCPU));
  counts_tensor =
      counts_tensor.to(self.device())
          .to(running_mean.defined() ? running_mean.dtype() : self.dtype());
  return cnnl_batch_norm_gather_stats_with_counts(
      self,
      mean,
      invstd,
      running_mean,
      running_var,
      momentum,
      epsilon,
      counts_tensor);
}

std::tuple<at::Tensor, at::Tensor> cnnl_batch_norm_gather_stats_with_counts(
    const at::Tensor& self,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& running_mean_opt,
    const c10::optional<at::Tensor>& running_var_opt,
    double momentum,
    double epsilon,
    const at::Tensor& counts) {
  const at::Tensor& running_mean =
      *at::borrow_from_optional_tensor(running_mean_opt);
  const at::Tensor& running_var =
      c10::value_or_else(running_var_opt, [] { return Tensor(); });

  TORCH_CHECK(
      mean.dim() == 2 && invstd.dim() == 2,
      "mean.dim() and invstd.dim() must equal 2, but got ",
      mean.dim(),
      " and ",
      invstd.dim());
  TORCH_CHECK(
      (!running_mean.defined() || running_mean.dim() == 1) &&
          (!running_var.defined() || running_var.dim() == 1) &&
          counts.dim() == 1,
      "running_mean.dim(), running_var.dim() and counts.dim() must equal 1");
  TORCH_CHECK(
      mean.scalar_type() == invstd.scalar_type() &&
          (!running_mean.defined() ||
           running_mean.scalar_type() == running_var.scalar_type()),
      "data type of mean must equal data type of var");
  TORCH_CHECK(
      !running_mean.defined() ||
          running_mean.scalar_type() == counts.scalar_type(),
      "data type of counts must equal data type of running_mean, but got ",
      running_mean.scalar_type(),
      " and ",
      counts.scalar_type());
  TORCH_CHECK(
      mean.scalar_type() == at::ScalarType::Float,
      "mean and invstd currently only support float data type, but got ",
      mean.scalar_type());

  auto momentum_f = c10::checked_convert<float, double>(momentum, "float");
  auto epsilon_f = c10::checked_convert<float, double>(epsilon, "float");
  constexpr c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous;
  auto mean_contiguous = cnnl_contiguous(mean, memory_format);
  auto invstd_contiguous = cnnl_contiguous(invstd, memory_format);
  auto running_mean_contiguous = running_mean.defined()
      ? cnnl_contiguous(running_mean, memory_format)
      : running_mean;
  auto running_var_contiguous = running_var.defined()
      ? cnnl_contiguous(running_var, memory_format)
      : running_var;
  auto counts_contiguous = cnnl_contiguous(counts, memory_format);

  auto scalar_type =
      running_mean.defined() ? running_mean.scalar_type() : self.scalar_type();
  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scalar_type,
      "batch_norm_update_stats_mlu",
      [&] {
        auto res = cnnl_batch_norm_gather_stats_with_counts_internal(
            mean_contiguous,
            invstd_contiguous,
            counts_contiguous,
            running_mean_contiguous,
            running_var_contiguous,
            momentum_f,
            epsilon_f);

        if (running_mean.defined() &&
            is_copy_necessary(running_mean, running_mean_contiguous)) {
          running_mean.copy_(running_mean_contiguous);
        }
        if (running_var.defined() &&
            is_copy_necessary(running_var, running_var_contiguous)) {
          running_var.copy_(running_var_contiguous);
        }

        return res;
      });
}

at::Tensor cnnl_batch_norm_elemt(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double epsilon) {
  int64_t dim = self.dim();
  auto memory_format = (3 < dim && dim < 6)
      ? get_channels_last_memory_format(dim)
      : LEGACY_CONTIGUOUS_MEMORY_FORMAT;
  auto output = at::empty_like(self, memory_format);
  cnnl_batch_norm_elemt_out(
      self, weight_opt, bias_opt, mean, invstd, epsilon, output);
  return output;
}

at::Tensor& cnnl_batch_norm_elemt_out(
    const at::Tensor& self,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& bias_opt,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double epsilon,
    at::Tensor& output) {
  const at::Tensor& weight = *at::borrow_from_optional_tensor(weight_opt);
  const at::Tensor& bias = *at::borrow_from_optional_tensor(bias_opt);

  TORCH_CHECK(self.numel() > 0, "currently do not support empty input as GPU");
  TORCH_CHECK(
      mean.scalar_type() == invstd.scalar_type() &&
          mean.scalar_type() == at::ScalarType::Float,
      "data type of mean and invstd must be float, but got ",
      mean.scalar_type(),
      " and ",
      invstd.scalar_type());
  TORCH_CHECK(
      !weight.defined() ||
          (weight.scalar_type() == bias.scalar_type() &&
           weight.scalar_type() == at::ScalarType::Float),
      "data type of weight and bias must be float, ",
      "but got ",
      weight.scalar_type(),
      " and ",
      bias.scalar_type());
  TORCH_CHECK(self.dim() >= 2, "input.dim() must be greater than 2");
  TORCH_CHECK(
      self.scalar_type() == output.scalar_type(),
      "data type of self and output must be equal, but got ",
      self.scalar_type(),
      " and ",
      output.scalar_type());
  TORCH_CHECK(
      mean.dim() == 1 && invstd.dim() == 1,
      "mean.dim() and invstd.dim() must equal 1, but got ",
      mean.dim(),
      " and ",
      invstd.dim());
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && bias.dim() == 1),
      "weight.dim() and bias.dim() must equal 1, but got ",
      weight.dim(),
      " and ",
      bias.dim());

  at::Tensor self_reshaped = self;
  // CNNL does not support input larger than 5D
  if (self.dim() > 5) {
    self_reshaped = self.reshape({self.size(0), self.size(1), -1});
  }
  auto need_squeeze = 3 == self_reshaped.dim();
  // Pytorch does not support 3D channel last currently
  if (need_squeeze) {
    self_reshaped = self_reshaped.unsqueeze(3);
  }
  int64_t dim = self_reshaped.dim();
  auto memory_format = dim > 3 ? get_channels_last_memory_format(dim)
                               : LEGACY_CONTIGUOUS_MEMORY_FORMAT;
  auto self_channels_last = cnnl_contiguous(self_reshaped, memory_format);
  resize_impl_mlu_(
      getMluTensorImpl(output),
      self_channels_last.sizes(),
      self_channels_last.strides());
  auto weight_contiguous = weight.defined()
      ? cnnl_contiguous(weight, c10::MemoryFormat::Contiguous)
      : weight;
  auto bias_contiguous = bias.defined()
      ? cnnl_contiguous(bias, c10::MemoryFormat::Contiguous)
      : bias;
  auto mean_contiguous = cnnl_contiguous(mean, c10::MemoryFormat::Contiguous);
  auto invstd_contiguous =
      cnnl_contiguous(invstd, c10::MemoryFormat::Contiguous);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_elementwise_mlu",
      [&] {
        cnnl_batch_norm_elemt_out_internal(
            output,
            self_channels_last,
            weight_contiguous,
            bias_contiguous,
            mean_contiguous,
            invstd_contiguous);
      });
  if (need_squeeze) {
    output.squeeze_(3);
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl_batch_norm_backward_reduce(
    const at::Tensor& self,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight_opt,
    bool input_g,
    bool weight_g,
    bool bias_g) {
  const at::Tensor& weight = *at::borrow_from_optional_tensor(weight_opt);

  TORCH_CHECK(input.numel() > 0, "currently do not support empty input as GPU");
  TORCH_CHECK(
      mean.scalar_type() == invstd.scalar_type() &&
          mean.scalar_type() == at::ScalarType::Float,
      "data type of mean and invstd must be float, but got ",
      mean.scalar_type(),
      " and ",
      invstd.scalar_type());
  TORCH_CHECK(
      !weight.defined() || weight.scalar_type() == at::ScalarType::Float,
      "data type of weight must be float, but got ",
      weight.scalar_type());
  TORCH_CHECK(input.dim() >= 2, "input.dim() must be greater than 2");
  TORCH_CHECK(
      input.sizes() == self.sizes(),
      "input.sizes() and grad_out.sizes() must be "
      "equal, but got ",
      input.sizes(),
      " and ",
      self.sizes());
  TORCH_CHECK(
      input.scalar_type() == self.scalar_type(),
      "data type of input and grad_out "
      "must be equal, but got ",
      input.scalar_type(),
      " and ",
      self.scalar_type());

  at::Tensor self_reshaped = self;
  at::Tensor input_reshaped = input;
  // CNNL does not support input larger than 5D
  if (self.dim() > 5) {
    self_reshaped = self.reshape({self.size(0), self.size(1), -1});
    input_reshaped = input.reshape({input.size(0), input.size(1), -1});
  }
  // Pytorch does not support 3D channel last currently
  if (self_reshaped.dim() == 3) {
    self_reshaped = self_reshaped.unsqueeze(3);
    input_reshaped = input_reshaped.unsqueeze(3);
  }
  int64_t dim = self_reshaped.dim();
  auto memory_format = dim > 3 ? get_channels_last_memory_format(dim)
                               : LEGACY_CONTIGUOUS_MEMORY_FORMAT;
  auto self_channels_last = cnnl_contiguous(self_reshaped, memory_format);
  auto input_channels_last = cnnl_contiguous(input_reshaped, memory_format);
  auto mean_contiguous = cnnl_contiguous(mean, c10::MemoryFormat::Contiguous);
  auto invstd_contiguous =
      cnnl_contiguous(invstd, c10::MemoryFormat::Contiguous);

  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_backward_reduce_mlu",
      [&] {
        return cnnl_batch_norm_backward_reduce_internal(
            self_channels_last,
            input_channels_last,
            mean_contiguous,
            invstd_contiguous,
            weight,
            input_g,
            weight_g,
            bias_g);
      });
}

at::Tensor cnnl_batch_norm_backward_elemt(
    const at::Tensor& self,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const c10::optional<at::Tensor>& weight_opt,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count) {
  const at::Tensor& weight = *at::borrow_from_optional_tensor(weight_opt);

  TORCH_CHECK(input.numel() > 0, "currently do not support empty input as GPU");
  TORCH_CHECK(
      mean.scalar_type() == invstd.scalar_type() &&
          mean.scalar_type() == at::ScalarType::Float,
      "data type of mean and invstd must be float, but got ",
      mean.scalar_type(),
      " and ",
      invstd.scalar_type());
  TORCH_CHECK(
      sum_dy.scalar_type() == sum_dy_xmu.scalar_type() &&
          sum_dy.scalar_type() == at::ScalarType::Float,
      "data type of sum_dy and sum_dy_xmu must be float, but got ",
      sum_dy.scalar_type(),
      " and ",
      sum_dy_xmu.scalar_type());
  TORCH_CHECK(
      !weight.defined() || weight.scalar_type() == at::ScalarType::Float,
      "data type of weight must be float, but got ",
      weight.scalar_type());
  TORCH_CHECK(input.dim() >= 2, "input.dim() must be greater than 2");
  TORCH_CHECK(
      input.sizes() == self.sizes(),
      "input.sizes() and grad_out.sizes() must be "
      "equal, but got ",
      input.sizes(),
      " and ",
      self.sizes());
  TORCH_CHECK(
      input.scalar_type() == self.scalar_type(),
      "data type of input and grad_out "
      "must be equal, but got ",
      input.scalar_type(),
      " and ",
      self.scalar_type());

  at::Tensor self_reshaped = self;
  at::Tensor input_reshaped = input;
  // CNNL does not support input larger than 5D
  if (self.dim() > 5) {
    self_reshaped = self.reshape({self.size(0), self.size(1), -1});
    input_reshaped = input.reshape({input.size(0), input.size(1), -1});
  }
  auto need_squeeze = 3 == self_reshaped.dim();
  // Pytorch does not support 3D channel last currently
  if (need_squeeze) {
    self_reshaped = self_reshaped.unsqueeze(3);
    input_reshaped = input_reshaped.unsqueeze(3);
  }
  int64_t dim = self_reshaped.dim();
  auto memory_format = dim > 3 ? get_channels_last_memory_format(dim)
                               : LEGACY_CONTIGUOUS_MEMORY_FORMAT;
  auto self_channels_last = cnnl_contiguous(self_reshaped, memory_format);
  auto input_channels_last = cnnl_contiguous(input_reshaped, memory_format);
  auto mean_contiguous = cnnl_contiguous(mean, c10::MemoryFormat::Contiguous);
  auto invstd_contiguous =
      cnnl_contiguous(invstd, c10::MemoryFormat::Contiguous);
  auto weight_contiguous = weight.defined()
      ? cnnl_contiguous(weight, c10::MemoryFormat::Contiguous)
      : weight;
  auto sum_dy_contiguous =
      cnnl_contiguous(sum_dy, c10::MemoryFormat::Contiguous);
  auto sum_dy_xmu_contiguous =
      cnnl_contiguous(sum_dy_xmu, c10::MemoryFormat::Contiguous);
  auto count_contiguous = cast_long_to_int_if_needed(
      cnnl_contiguous(count, c10::MemoryFormat::Contiguous));

  return AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "batch_norm_backward_elemt_mlu",
      [&] {
        auto res = cnnl_batch_norm_backward_elemt_internal(
            self_channels_last,
            input_channels_last,
            mean_contiguous,
            invstd_contiguous,
            weight_contiguous,
            sum_dy_contiguous,
            sum_dy_xmu_contiguous,
            count_contiguous);
        if (need_squeeze) {
          res.squeeze_(3);
        }

        return res;
      });
}

} // namespace ops
} // namespace torch_mlu
