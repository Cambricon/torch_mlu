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

#include <torch/autograd.h>
#include "aten/utils/dispatch.h"
#include <ATen/native/Pool.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_max_unpool2d_out(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_size) {
  TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long ||
          indices.scalar_type() == at::ScalarType::Int ||
          indices.scalar_type() == at::ScalarType::Short,
      "elements in indices should be int16/int32/int64 but got: ",
      indices.scalar_type());
  auto oheight = output_size[0];
  auto owidth = output_size[1];

  for (int64_t i = 1; i < self.ndimension(); ++i) {
    TORCH_CHECK(
        self.size(i) > 0,
        "max_unpool2d_forward_out_mlu(): ",
        "Expected input to have non-zero size for non-batch dimensions, but got ",
        self.sizes(),
        " with dimension ",
        i,
        " being empty.");
  }

  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");
  TORCH_CHECK(
      (self.ndimension() == 3 || self.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor");
  TORCH_CHECK(
      self.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "element in self should be floating types");
  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  if (self.ndimension() == 3) {
    int64_t numChannels = self_contiguous.size(0);
    out.resize_({numChannels, oheight, owidth}, memory_format);
  } else {
    int64_t numBatch = self_contiguous.size(0);
    int64_t numChannels = self_contiguous.size(1);
    out.resize_({numBatch, numChannels, oheight, owidth}, memory_format);
  }
  out.zero_();
  AT_DISPATCH_MLU_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "cnnl_max_unpool2d_forward_out", [&] {
        cnnl_max_unpool2d_internal(
            out,
            self_contiguous,
            indices_contiguous,
            kernel_size,
            stride,
            padding);
      });
  return out;
}

at::Tensor cnnl_max_unpool2d(
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  return cnnl_max_unpool2d_out(
      output, self, indices, kernel_size, stride, padding, output_size);
}

at::Tensor cnnl_max_unpool2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& grad_input) {
  if (grad_input.sizes() != self.sizes()) {
    grad_input.resize_as_(self);
  }
  grad_input.zero_();
  auto memory_format = get_channels_last_memory_format(self.dim());
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);
  auto grad_input_contiguous = cnnl_contiguous(grad_input, memory_format);
  AT_DISPATCH_MLU_FLOATING_TYPES_AND_HALF(
      self.scalar_type(), "cnnl_max_unpool2d_backward_out", [&] {
        cnnl_max_unpool2d_backward_internal(
            grad_output_contiguous,
            self_contiguous,
            indices_contiguous,
            kernel_size,
            stride,
            padding,
            grad_input_contiguous);
        if (is_copy_necessary(grad_input, grad_input_contiguous)) {
          grad_input.copy_(grad_input_contiguous);
        }
      });
  return grad_input;
}

at::Tensor cnnl_max_unpool2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_size) {
  auto grad_input = at::empty_like(self);
  return cnnl_max_unpool2d_backward_out(
      grad_output, self, indices, kernel_size, stride, padding, grad_input);
}

class MAXUnpool2dFunction
    : public torch::autograd::Function<MAXUnpool2dFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& self,
      const at::Tensor& indices,
      at::IntArrayRef kernel_size,
      at::IntArrayRef stride,
      at::IntArrayRef padding,
      at::IntArrayRef output_size) {
    at::AutoDispatchBelowADInplaceOrView g;
    ctx->save_for_backward({self, indices});
    ctx->saved_data["kernel_size"] = kernel_size;
    ctx->saved_data["stride"] = stride;
    ctx->saved_data["padding"] = padding;
    ctx->saved_data["output_size"] = output_size;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("torch_mlu::max_unpool2d", "")
                         .typed<decltype(cnnl_max_unpool2d)>();
    auto result =
        op.call(self, indices, kernel_size, stride, padding, output_size);
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto self = saved[0];
    auto indices = saved[1];
    std::vector<int64_t> kernel_size_vector =
        ctx->saved_data["kernel_size"].toIntVector();
    at::IntArrayRef kernel_size = kernel_size_vector;
    std::vector<int64_t> stride_vector =
        ctx->saved_data["stride"].toIntVector();
    at::IntArrayRef stride = stride_vector;
    std::vector<int64_t> padding_vector =
        ctx->saved_data["padding"].toIntVector();
    at::IntArrayRef padding = padding_vector;
    std::vector<int64_t> output_size_vector =
        ctx->saved_data["output_size"].toIntVector();
    at::IntArrayRef output_size = output_size_vector;
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torch_mlu::max_unpool2d_backward", "")
            .typed<decltype(cnnl_max_unpool2d_backward)>();
    auto result = op.call(
        grad_outputs[0],
        self,
        indices,
        kernel_size,
        stride,
        padding,
        output_size);
    return {
        result,
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable(),
        torch::autograd::Variable()};
  }
};

at::Tensor cnnl_max_unpool2d_autograd(
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef output_size) {
  return MAXUnpool2dFunction::apply(
      self, indices, kernel_size, stride, padding, output_size)[0];
}

} // namespace ops
} // namespace torch_mlu
