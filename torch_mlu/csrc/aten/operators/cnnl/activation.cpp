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
#include "aten/DispatchStub.h"
#include "ATen/native/Activation.h"
#include "ATen/native/UnaryOps.h"
#include "ATen/native/BinaryOps.h"
#include "aten/TensorIteratorBridge.h"
#include "ATen/CPUGeneratorImpl.h"
#include "ATen/core/DistributionsHelper.h"
#include "aten/utils/dispatch.h"

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

namespace torch_mlu {
namespace ops {
using namespace at::native;
/***************************************relu****************************************/
// aten::relu will dispatch to aten::clamp_min
// ref:
// https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/native/Activation.cpp#L443
at::Tensor cnnl_relu(const at::Tensor& self) {
  return at::native::relu(self);
}
// aten::relu_ will dispatch to aten::clamp_min_
// ref:
// https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/native/Activation.cpp#L448
at::Tensor& cnnl_relu_(at::Tensor& self) {
  return at::native::relu_(self);
}

/***************************************softplus****************************************/
void softplus_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& beta_,
    const at::Scalar& threshold_) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto input = iter.input(0);
  cnnl_softplus_internal(output, input, beta_, threshold_);
}

void softplus_backward_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& beta_,
    const at::Scalar& threshold_) {
  if (iter.numel() == 0)
    return;
  auto result = iter.output(0);
  auto grad_output = iter.input(0);
  auto self = iter.input(1);
  cnnl_softplus_backward_internal(grad_output, self, beta_, threshold_, result);
}

/***************************************tanh****************************************/
void tanh_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_activation_internal(output, self, CNNL_ACTIVATION_TANH);
}

void tanh_backward_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(1);
  auto grad = iter.input(0);
  cnnl_activation_backward_internal(output, self, grad, CNNL_ACTIVATION_TANH);
}
/***************************************gelu****************************************/
void GeluMLUKernelImpl(at::TensorIteratorBase& iter, GeluType approximate) {
  auto output = iter.output(0);
  bool approximate_value = approximate == GeluType::Tanh ? true : false;
  cnnl_activation_internal(
      output,
      iter.input(0),
      CNNL_ACTIVATION_GELU,
      0,
      0.0,
      0.0,
      0.0,
      approximate_value);
}

void GeluBackwardMLUKernelImpl(
    at::TensorIteratorBase& iter,
    GeluType approximate) {
  auto output = iter.output(0);
  bool approximate_value = approximate == GeluType::Tanh ? true : false;
  cnnl_activation_backward_internal(
      output,
      iter.input(1),
      iter.input(0),
      CNNL_ACTIVATION_GELU,
      0,
      0.0,
      0.0,
      0.0,
      approximate_value);
}

TORCH_IMPL_FUNC(gelu_out_mlu)
(const Tensor& self, c10::string_view approximate, const Tensor& result) {
  GeluMLUKernelImpl(*this, get_gelutype_enum(approximate));
}

TORCH_IMPL_FUNC(gelu_backward_out_mlu)
(const Tensor& grad,
 const Tensor& self,
 c10::string_view approximate,
 const Tensor& grad_input) {
  GeluBackwardMLUKernelImpl(*this, get_gelutype_enum(approximate));
}
/***************************************glu****************************************/
void glu_mlu_kernel_impl(at::TensorIteratorBase& iter, int64_t dim) {
  auto& out = iter.output(0);
  auto& input = iter.input(0);
  auto memory_format = input.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(input, memory_format);
  auto out_contiguous = cnnl_contiguous(out, memory_format);
  cnnl_activation_internal(
      out_contiguous, self_contiguous, CNNL_ACTIVATION_GLU, dim);
  if (is_copy_necessary(out_contiguous, out)) {
    out.copy_(out_contiguous);
  }
}

void glu_backward_mlu_kernel_impl(at::TensorIteratorBase& iter, int64_t dim) {
  auto& grad_input = iter.output(0);
  auto& input = iter.input(0);
  auto& grad_out = iter.input(1);
  auto memory_format = input.suggest_memory_format();
  auto grad_input_contiguous = cnnl_contiguous(grad_input, memory_format);
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto grad_out_contiguous = cnnl_contiguous(grad_out, memory_format);
  cnnl_activation_backward_internal(
      grad_input_contiguous,
      input_contiguous,
      grad_out_contiguous,
      CNNL_ACTIVATION_GLU,
      dim);
  if (is_copy_necessary(grad_input_contiguous, grad_input)) {
    grad_input.copy_(grad_input_contiguous);
  }
}
// This is also structured op, but not use the way of structured.
// For the Implement kernel of 'glu_stub' is different from cnnl kernel,
at::Tensor& cnnl_glu_out(const at::Tensor& self, int64_t dim, at::Tensor& out) {
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(
      nIn % 2 == 0,
      "Halving dimension must be even, but dimension ",
      wrap_dim,
      " is size ",
      nIn);
  const int64_t selfSize = nIn / 2;
  auto newSizes = self.sizes().vec();
  newSizes[wrap_dim] = selfSize;
  out.resize_(newSizes);

  // Build TensorIterator.
  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .declare_static_shape(newSizes)
                  .add_output(out)
                  .add_input(self)
                  .build();
  glu_mlu_kernel_impl(iter, wrap_dim);
  return out;
}

at::Tensor cnnl_glu_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    int64_t dim) {
  auto grad_input = at::empty({0}, self.options());
  return cnnl_glu_backward_out(grad_output, self, dim, grad_input);
}

at::Tensor& cnnl_glu_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    int64_t dim,
    at::Tensor& grad_input) {
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = at::maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(
      nIn % 2 == 0,
      "Halving dimension must be even, but dimension ",
      wrap_dim,
      " is size ",
      nIn);
  grad_input.resize_as_(self);
  const int64_t selfSize = nIn / 2;
  auto newSizes = self.sizes().vec();
  newSizes[wrap_dim] = selfSize;

  // Build TensorIterator.
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(grad_output)
                  .resize_outputs(false)
                  .declare_static_shape(newSizes)
                  .build();
  if (iter.numel() == 0) {
    return grad_input;
  }
  glu_backward_mlu_kernel_impl(iter, wrap_dim);
  return grad_input;
}
/***************************************hardswish****************************************/
at::Tensor cnnl_hardswish(const at::Tensor& self) {
  return at::native::hardswish(self);
}

at::Tensor& cnnl_hardswish_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::hardswish_out(self, out);
}

at::Tensor& cnnl_hardswish_(at::Tensor& self) {
  return at::native::hardswish_(self);
}

at::Tensor cnnl_hardswish_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  return at::native::hardswish_backward(grad_output, self);
}

void hardswish_mlu_kernel(at::TensorIterator& iter) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "hardswish");
  auto output = iter.output(0);
  cnnl_activation_internal(output, iter.input(0), CNNL_ACTIVATION_HARDSWISH);
  iter_bridge.cast_outputs(iter);
}

void hardswish_backward_mlu_kernel(at::TensorIterator& iter) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "hardswish_backward");
  auto output = iter.output(0);
  cnnl_activation_backward_internal(
      output, iter.input(1), iter.input(0), CNNL_ACTIVATION_HARDSWISH);
  iter_bridge.cast_outputs(iter);
}
/***************************************hardtanh****************************************/
// native hardtanh ops will dispatch to clamp kernel
at::Tensor cnnl_hardtanh(
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val) {
  return at::native::hardtanh(self, min_val, max_val);
}
at::Tensor& cnnl_hardtanh_out(
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val,
    at::Tensor& out) {
  return at::native::hardtanh_out(self, min_val, max_val, out);
}

at::Tensor& cnnl_hardtanh_(
    at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val) {
  return at::native::hardtanh_(self, min_val, max_val);
}

void hardtanh_backward_mlu_kernel(
    at::TensorIterator& iter,
    const at::Scalar& min_val,
    const at::Scalar& max_val) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "hardtanh");
  auto output = iter.output(0);
  cnnl_hardtanh_backward_internal(
      output, iter.input(1), iter.input(0), min_val, max_val);
  iter_bridge.cast_outputs(iter);
}

at::Tensor cnnl_hardtanh_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val) {
  return at::native::hardtanh_backward(grad_output, self, min_val, max_val);
}

at::Tensor& cnnl_hardtanh_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val,
    at::Tensor& grad_input) {
  return at::native::hardtanh_backward_out(
      grad_output, self, min_val, max_val, grad_input);
}
/***************************************hardsigmoid****************************************/
void hardsigmoid_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_activation_internal(output, iter.input(0), CNNL_ACTIVATION_HARDSIGMOID);
}
void hardsigmoid_backward_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_activation_backward_internal(
      output, iter.input(1), iter.input(0), CNNL_ACTIVATION_HARDSIGMOID);
}
/***************************************log_sigmoid****************************************/
void launch_log_sigmoid_forward_mlu_kernel(at::TensorIteratorBase& iter) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "log_sigmoid");
  auto output = iter.output(0);
  cnnl_activation_internal(output, iter.input(0), CNNL_ACTIVATION_LOGSIGMOID);
  iter_bridge.cast_outputs(iter);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_log_sigmoid_forward_out(
    const at::Tensor& self,
    at::Tensor& output,
    at::Tensor& buffer) {
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  auto iter =
      at::TensorIteratorConfig().add_output(output).add_input(self).build();
  launch_log_sigmoid_forward_mlu_kernel(iter);
  return std::forward_as_tuple(output, buffer);
}

std::tuple<at::Tensor, at::Tensor> cnnl_log_sigmoid_forward(
    const at::Tensor& self) {
  auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto buffer = at::empty({0}, self.options());
  cnnl_log_sigmoid_forward_out(self, output, buffer);
  return std::forward_as_tuple(output, buffer);
}

void log_sigmoid_backward_mlu_kernel(at::TensorIterator& iter) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "log_sigmoid");
  auto output = iter.output(0);
  cnnl_activation_backward_internal(
      output, iter.input(0), iter.input(1), CNNL_ACTIVATION_LOGSIGMOID);
  iter_bridge.cast_outputs(iter);
}

at::Tensor cnnl_log_sigmoid_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& buffer) {
  auto grad_input = at::empty_like(grad_output);
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(grad_output)
                  .build();
  log_sigmoid_backward_mlu_kernel(iter);
  return iter.output();
}

at::Tensor& cnnl_log_sigmoid_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& buffer,
    at::Tensor& grad_input) {
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(grad_output)
                  .build();
  log_sigmoid_backward_mlu_kernel(iter);
  return grad_input;
}
/***************************************sigmoid****************************************/
void sigmoid_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_activation_internal(output, self, CNNL_ACTIVATION_SIGMOID);
}
void sigmoid_backward_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_activation_backward_internal(
      output, iter.input(1), iter.input(0), CNNL_ACTIVATION_SIGMOID);
}
/***************************************elu****************************************/
void elu_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale) {
  cnnlActivationMode_t mode = CNNL_ACTIVATION_ELU_V2;
  if (std::abs(SELU_SCALE - scale.to<double>()) <= 1e-8 &&
      std::abs(SELU_ALPHA - alpha.to<double>()) <= 1e-8) {
    mode = CNNL_ACTIVATION_SELU;
  }
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_activation_internal(output, self, mode, 0, alpha, scale, input_scale);
}
void elu_backward_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& alpha,
    const at::Scalar& scale,
    const at::Scalar& input_scale,
    bool is_result) {
  auto output = iter.output(0);
  cnnl_activation_backward_internal(
      output,
      iter.input(1),
      iter.input(0),
      CNNL_ACTIVATION_ELU_V2,
      0,
      alpha,
      scale,
      input_scale,
      is_result);
}
/***************************************leaky_relu****************************************/
void leaky_relu_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& negval_) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_activation_internal(output, self, CNNL_ACTIVATION_LEAKYRELU, 0, negval_);
}
void leaky_relu_backward_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& negval_) {
  auto output = iter.output(0);
  cnnl_activation_backward_internal(
      output,
      iter.input(0),
      iter.input(1),
      CNNL_ACTIVATION_LEAKYRELU,
      0,
      negval_);
}
/***************************************silu****************************************/
void silu_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_activation_internal(output, self, CNNL_ACTIVATION_SILU);
}

void silu_backward_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_activation_backward_internal(
      output, iter.input(1), iter.input(0), CNNL_ACTIVATION_SILU);
}
/***************************************threshold****************************************/
void threshold_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& threshold,
    const at::Scalar& value) {
  auto output = create_int_tensor_if_needed(iter.output(0));
  auto self = cast_long_to_int_if_needed(iter.input(0));
  cnnl_threshold_internal(output, self, threshold, value);
  cast_int_to_long_if_needed(output, iter.output(0));
}

// This is also structured op, but not use the way of structured.
// For the two Implement kernel of 'threshold' and 'threshold_backward' not
// distinguish in torch, they are all dispatch through 'threshold_stub'.
at::Tensor& cnnl_threshold_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold,
    at::Tensor& grad_input) {
  auto iter = at::TensorIteratorConfig()
                  .set_check_mem_overlap(
                      false) // threshold is idempotent, so overlap is okay
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(grad_output) // other
                  .allow_cpu_scalars(true)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "threshold_backward");
  auto result = create_int_tensor_if_needed(iter.output(0));
  auto input_0 = cast_long_to_int_if_needed(iter.input(0));
  auto input_1 = cast_long_to_int_if_needed(iter.input(1));
  cnnl_threshold_backward_internal(result, input_1, input_0, threshold);
  cast_int_to_long_if_needed(result, iter.output(0));
  iter_bridge.cast_outputs(iter);
  return grad_input;
}

/***************************************rrelu****************************************/
// aten::rrelu will dispatch to aten::rrelu_with_noise
static void rrelu_out_mlu(
    at::Tensor& output,
    const at::Tensor& self,
    at::Scalar lower,
    at::Scalar upper,
    bool training,
    std::optional<at::Generator> generator) {
  output.resize_as_(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "cnnl_rrelu",
      [&] {
        at::Scalar negative_slope;
        if (training) {
          auto gen = at::get_generator_or_default<at::CPUGeneratorImpl>(
              generator, at::detail::getDefaultCPUGenerator());
          at::uniform_real_distribution<double> uniform(
              lower.to<scalar_t>(), upper.to<scalar_t>());
          const scalar_t r = (scalar_t)uniform(gen);
          negative_slope = at::Scalar(r);
        } else {
          negative_slope = (lower.to<float>() + upper.to<float>()) / 2;
        }
        at::leaky_relu_out(output, self, negative_slope);
      });
}

at::Tensor cnnl_rrelu_with_noise(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    std::optional<at::Generator> generator) {
  at::Tensor output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return cnnl_rrelu_with_noise_out(
      self, noise, lower, upper, training, generator, output);
}

at::Tensor& cnnl_rrelu_with_noise_out(
    const at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  if (self.numel() == 0) {
    return out;
  }
  at::TensorArg self_arg{self, "self", 1}, noise_arg{noise, "noise", 2},
      output_arg{out, "out", 3};
  checkAllSameMLU(
      "cnnl_rrelu_with_noise_out", {self_arg, noise_arg, output_arg});
  rrelu_out_mlu(out, self, lower, upper, training, generator);
  return out;
}
at::Tensor& cnnl_rrelu_with_noise_(
    at::Tensor& self,
    const at::Tensor& noise,
    const at::Scalar& lower,
    const at::Scalar& upper,
    bool training,
    std::optional<at::Generator> generator) {
  return cnnl_rrelu_with_noise_out(
      self, noise, lower, upper, training, generator, self);
}
/***************************************softshrink****************************************/
void softshrink_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& value) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_activation_internal(output, self, CNNL_ACTIVATION_SOFTSHRINK, 0, value);
  iter.cast_outputs();
}
void shrink_backward_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& value) {
  auto output = iter.output(0);
  // fix hardshrink mode, same autograd formula for both softshrink and
  // hardshrink
  cnnl_activation_backward_internal(
      output,
      iter.input(1),
      iter.input(0),
      CNNL_ACTIVATION_HARDSHRINK,
      0,
      value);
  iter.cast_outputs();
}
/***************************************hardshrink****************************************/
void hardshrink_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& value) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_activation_internal(output, self, CNNL_ACTIVATION_HARDSHRINK, 0, value);
  iter.cast_outputs();
}
/**********************************REGISTER_PRIVATEUSE1_DISPATCH**************************/
REGISTER_PRIVATEUSE1_DISPATCH(softplus_stub, &softplus_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    softplus_backward_stub,
    &softplus_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(tanh_stub, &tanh_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(tanh_backward_stub, &tanh_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(hardswish_stub, &hardswish_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    hardswish_backward_stub,
    &hardswish_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    hardtanh_backward_stub,
    &hardtanh_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(hardsigmoid_stub, &hardsigmoid_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    hardsigmoid_backward_stub,
    &hardsigmoid_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(sigmoid_stub, &sigmoid_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    sigmoid_backward_stub,
    &sigmoid_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(elu_stub, &elu_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(elu_backward_stub, &elu_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(leaky_relu_stub, &leaky_relu_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    leaky_relu_backward_stub,
    &leaky_relu_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(silu_stub, &silu_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(silu_backward_stub, &silu_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(threshold_stub, &threshold_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(softshrink_stub, &softshrink_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    shrink_backward_stub,
    &shrink_backward_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(hardshrink_stub, &hardshrink_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
