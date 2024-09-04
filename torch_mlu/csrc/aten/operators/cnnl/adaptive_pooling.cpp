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
#include "c10/core/ScalarType.h"

namespace torch_mlu {
namespace ops {

at::Tensor create_out_adaptive_pooling(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  if (strides.empty()) {
    return at::empty(sizes, options);
  } else {
    return at::empty_strided(sizes, strides, options);
  }
}

void resize_out_adaptive_pooling(
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
  // MLU side don't check the resized status, cause CATCH always need contiguous
  // output tensor.
  if (!strides.empty()) {
    TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
    // TODO: avoid the redispatch here
    out.as_strided_(sizes, strides);
  } else if (options.memory_format_opt().has_value()) {
    out.unsafeGetTensorImpl()->empty_tensor_restride(
        *options.memory_format_opt());
  }
}

// cnnl adaptive_avg_pool3d and adaptive_max_pool2d has kernel size limitation.
void check_cnnl_limitation(
    const char* func_name,
    const at::Tensor& self,
    const int output_H,
    const int output_W) {
  if (output_H == 0 || output_W == 0)
    return;
  int H_limit = self.size(-2) / output_H + 2;
  int W_limit = self.size(-1) / output_W + 2;
  int limit_size = H_limit * W_limit;
  TORCH_CHECK(
      limit_size <= 3582,
      "The internal kernel size for " + std::string(func_name) +
          " should be "
          "smaller than 3582, while this kernel size is ",
      limit_size);
}

at::Tensor& cnnl_adaptive_avg_pool2d_out(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    at::Tensor& output) {
  at::TensorArg input_arg{self, "input", 1}, output_arg{output, "output", 2};
  checkAllSameMLU("cnnl_adaptive_avg_pool2d_out", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
  int64_t ndim = self.dim();
  TORCH_CHECK(
      (ndim == 3 || ndim == 4),
      "adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got: ",
      self.sizes());

  for (const auto i : {-2, -1}) {
    TORCH_CHECK(
        self.size(i) > 0,
        "adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        self.sizes(),
        " with dimension ",
        i + ndim,
        " being "
        "empty");
  }

  at::Tensor self_4d = self;
  if (self.dim() == 3) {
    self_4d = self.unsqueeze(0);
  }

  resize_impl_mlu_(
      getMluTensorImpl(output),
      {self_4d.size(0), self_4d.size(1), output_size[0], output_size[1]},
      c10::nullopt);
  if (output.numel() == 0) {
    return self.dim() == 3 ? output.squeeze_(0) : output;
  }

  auto memory_format = get_channels_last_memory_format(self_4d.dim());
  getMluTensorImpl(output)->empty_tensor_restride(memory_format);
  auto self_contiguous = cnnl_contiguous(self_4d, memory_format);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "cnnl_adaptive_avg_pool2d_out",
      [&] {
        cnnl_adaptive_avg_pool_internal(output, self_contiguous, output_size);
        if (self.dim() == 3) {
          output.squeeze_(0);
        }
      });

  return output;
}

at::Tensor cnnl__adaptive_avg_pool2d(
    const at::Tensor& self,
    at::IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  cnnl_adaptive_avg_pool2d_out(self, output_size, output);
  return output;
}

at::Tensor cnnl__adaptive_avg_pool2d_backward(
    const at::Tensor& gradOutput,
    const at::Tensor& self) {
  auto input = self.ndimension() == 3 ? self.unsqueeze(0) : self;
  auto grad_output =
      self.ndimension() == 3 ? gradOutput.unsqueeze(0) : gradOutput;

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto input_contiguous = cnnl_contiguous(input, memory_format);

  auto gradInput = at::empty_like(input, memory_format);
  if (gradInput.numel() == 0) {
    return self.dim() == 3 ? gradInput.squeeze_(0) : gradInput;
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "cnnl__adaptive_avg_pool2d_backward",
      [&] {
        cnnl_adaptive_avg_pool_backward_internal(
            gradInput, grad_output_contiguous, input_contiguous);
        if (self.ndimension() == 3) {
          gradInput.squeeze_(0);
        }
      });

  return gradInput;
}

at::Tensor& cnnl_adaptive_avg_pool3d_out(
    const at::Tensor& input_,
    at::IntArrayRef output_size,
    at::Tensor& output) {
  at::TensorArg output_arg{output, "output", 1};
  at::TensorArg input_arg{input_, "input_", 2};

  checkAllSameMLU("cnnl_adaptive_avg_pool3d_out", {output_arg, input_arg});

  for (int64_t i = 1; i < input_.ndimension(); i++) {
    TORCH_CHECK(
        input_.size(i) > 0,
        "cnnl_adaptive_avg_pool3d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input_.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      (input_.ndimension() == 4 || input_.ndimension() == 5),
      "cnnl_adaptive_avg_pool3d(): Expected 4D or 5D tensor, but got ",
      input_.sizes());

  // the jit sometimes passes output_size.size() == 1
  TORCH_CHECK(
      output_size.size() == 1 || output_size.size() == 3,
      "adaptive_avg_pool3d: internal error: output_size.size() must be 1 or 3");

  /* output sizes */
  auto osizeT = output_size[0];
  auto osizeH = output_size[1];
  auto osizeW = output_size[2];

  int64_t sizeD;

  if (input_.ndimension() == 4) {
    sizeD = input_.size(0);

    output.resize_({1, sizeD, osizeT, osizeH, osizeW});

  } else {
    int64_t sizeB = input_.size(0);
    sizeD = input_.size(1);

    output.resize_({sizeB, sizeD, osizeT, osizeH, osizeW});
  }

  if (output.numel() == 0) {
    return input_.dim() == 4 ? output.squeeze_(0) : output;
  }

  const Tensor& input = input_.ndimension() == 4 ? input_.unsqueeze(0) : input_;
  // check_cnnl_limitation(__func__, input, output_size[1], output_size[2]);
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  getMluTensorImpl(output)->empty_tensor_restride(memory_format);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "cnnl_adaptive_avg_pool3d_out",
      [&] {
        cnnl_adaptive_avg_pool_internal(output, input_contiguous, output_size);
        if (input_.dim() == 4) {
          output.squeeze_(0);
        }
      });
  return output;
}

at::Tensor cnnl__adaptive_avg_pool3d(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  cnnl_adaptive_avg_pool3d_out(input, output_size, output);
  return output;
}

at::Tensor& cnnl_adaptive_avg_pool3d_backward_out(
    const at::Tensor& gradOutput,
    const at::Tensor& self,
    at::Tensor& grad_input) {
  at::TensorArg grad_input_arg{grad_input, "gradInput", 1};
  at::TensorArg grad_output_arg{gradOutput, "gradOutput_", 2};
  at::TensorArg input_arg{self, "input", 3};

  checkAllSameMLU(
      "cnnl_adaptive_avg_pool3d_out",
      {grad_input_arg, grad_output_arg, input_arg});
  if (grad_input.numel() == 0) {
    grad_input.resize_as_(self);
    return grad_input;
  }
  check_cnnl_limitation(
      __func__, self, gradOutput.size(-2), gradOutput.size(-1));

  auto input = self.ndimension() == 4 ? self.unsqueeze(0) : self;
  auto grad_output =
      self.ndimension() == 4 ? gradOutput.unsqueeze(0) : gradOutput;
  grad_input.resize_as_(input);
  grad_input.zero_();

  auto memory_format = get_channels_last_memory_format(input.dim());
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  getMluTensorImpl(grad_input)->empty_tensor_restride(memory_format);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "cnnl_adaptive_avg_pool3d_backward_out",
      [&] {
        cnnl_adaptive_avg_pool_backward_internal(
            grad_input, grad_output_contiguous, input_contiguous);
        if (self.ndimension() == 4) {
          grad_input.squeeze_(0);
        }
      });
  return grad_input;
}

at::Tensor cnnl__adaptive_avg_pool3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  cnnl_adaptive_avg_pool3d_backward_out(grad_output, input, gradInput);
  return gradInput;
}

std::tuple<int64_t, int64_t, int64_t, int64_t> adaptive_max_pool2d_pre_compute(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  int ndim = input.ndimension();
  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "adaptive_max_pool2d(): Expected 3D or 4D tensor, but got: ",
      input.sizes());
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_max_pool2d(): internal error: output_size.size() must be 2");

  int dimH = 1;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  if (input.ndimension() == 4) {
    sizeB = input.size(0);
    dimH++;
  }

  sizeD = input.size(dimH - 1);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];
  return std::make_tuple(sizeB, sizeD, osizeH, osizeW);
}

std::tuple<at::Tensor, at::Tensor> cnnl_adaptive_max_pool2d(
    const at::Tensor& input,
    at::IntArrayRef output_size) {
  int64_t sizeB, sizeD, osizeH, osizeW;
  auto precompute_results = adaptive_max_pool2d_pre_compute(input, output_size);
  std::tie(sizeB, sizeD, osizeH, osizeW) = precompute_results;
  at::Tensor output = create_out_adaptive_pooling(
      {sizeB, sizeD, osizeH, osizeW},
      {},
      input.options().memory_format(at::MemoryFormat::ChannelsLast));
  at::Tensor indices = create_out_adaptive_pooling(
      {sizeB, sizeD, osizeH, osizeW},
      {},
      input.options()
          .memory_format(at::MemoryFormat::ChannelsLast)
          .dtype(at::kLong));
  return cnnl_adaptive_max_pool2d_out(input, output_size, output, indices);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_adaptive_max_pool2d_out(
    const at::Tensor& input,
    at::IntArrayRef output_size,
    at::Tensor& output,
    at::Tensor& indices) {
  int64_t sizeB, sizeD, osizeH, osizeW;
  auto precompute_results = adaptive_max_pool2d_pre_compute(input, output_size);
  std::tie(sizeB, sizeD, osizeH, osizeW) = precompute_results;
  resize_out_adaptive_pooling(
      output,
      {sizeB, sizeD, osizeH, osizeW},
      {},
      input.options().memory_format(at::MemoryFormat::ChannelsLast));
  resize_out_adaptive_pooling(
      indices,
      {sizeB, sizeD, osizeH, osizeW},
      {},
      input.options()
          .memory_format(at::MemoryFormat::ChannelsLast)
          .dtype(at::kLong));

  at::TensorArg output_arg{output, "output", 1};
  at::TensorArg indices_arg{indices, "indices", 2};
  at::TensorArg input_arg{input, "input", 3};

  checkAllSameMLU(
      "adaptive_max_pool2d_out_mlu", {output_arg, indices_arg, input_arg});
  if (input.numel() == 0) {
    if (input.ndimension() == 3) {
      output.squeeze(0);
      indices.squeeze(0);
    }
    return std::forward_as_tuple(output, indices);
  }

  auto self_4d = input.ndimension() == 3 ? input.unsqueeze(0) : input;

  check_cnnl_limitation(
      "adaptive_max_pool2d_out_mlu", self_4d, output_size[0], output_size[1]);

  auto memory_format = get_channels_last_memory_format(self_4d.dim());
  auto self_contiguous = cnnl_contiguous(self_4d, memory_format);
  auto output_contiguous = cnnl_contiguous(output, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "adaptive_max_pool2d_out_mlu",
      [&] {
        cnnl_adaptive_max_pool2d_internal(
            output_contiguous,
            indices_contiguous,
            self_contiguous,
            output_size);
        if (is_copy_necessary(output, output_contiguous)) {
          output.copy_(output_contiguous);
        }

        if (is_copy_necessary(indices, indices_contiguous)) {
          indices.copy_(indices_contiguous);
        }
        if (input.ndimension() == 3) {
          output.squeeze_(0);
          indices.squeeze_(0);
        }
      });
  return std::forward_as_tuple(output, indices);
}

at::Tensor cnnl_adaptive_max_pool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices) {
  at::Tensor input_ = input.dim() == 3 ? at::unsqueeze(input, 0) : input;
  at::Tensor gradInput = create_out_adaptive_pooling(
      input_.sizes(),
      {},
      input_.options().memory_format(at::MemoryFormat::ChannelsLast));
  return cnnl_adaptive_max_pool2d_backward_out(
      grad_output, input, indices, gradInput);
}

at::Tensor& cnnl_adaptive_max_pool2d_backward_out(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    Tensor& gradInput) {
  int64_t ndim = gradOutput.ndimension();
  TORCH_CHECK(
      ndim == 3 || ndim == 4,
      "adaptive_max_pooling2d_backward(): Expected 3D or 4D grad_output, but got: ",
      gradOutput.sizes());
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        gradOutput.size(i) > 0,
        "adaptive_max_pooling2d_backward(): Expected grad_output to have non-zero size for non-batch dimensions, "
        "but grad_output has sizes ",
        gradOutput.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      input.dtype() == gradOutput.dtype(),
      "expected dtype ",
      input.dtype(),
      " for `grad_output` but got dtype ",
      gradOutput.dtype());
  auto input_4d = input.ndimension() == 3 ? input.unsqueeze(0) : input;
  resize_out_adaptive_pooling(
      gradInput,
      input_4d.sizes(),
      {},
      input_4d.options().memory_format(at::MemoryFormat::ChannelsLast));
  at::TensorArg grad_input_arg{gradInput, "gradInput", 1};
  at::TensorArg grad_output_arg{gradOutput, "gradOutput", 2};
  at::TensorArg input_arg{input, "input", 3};
  at::TensorArg indices_arg{indices, "indices", 4};

  checkAllSameMLU(
      __func__, {grad_input_arg, grad_output_arg, input_arg, indices_arg});

  if (gradOutput.numel() == 0) {
    if (input.ndimension() == 3) {
      gradInput.squeeze_(0);
    }
    return gradInput;
  }
  auto gradOutput_4d =
      input.ndimension() == 3 ? gradOutput.unsqueeze(0) : gradOutput;
  auto indices_4d = input.ndimension() == 3 ? indices.unsqueeze(0) : indices;

  // TODO(CNNLCORE-11573): remove this when cnnl support int32 index for half
  // dtype.
  if (input.scalar_type() == at::kHalf ||
      input.scalar_type() == at::kBFloat16) {
    indices_4d = indices_4d.to(at::kShort);
  } else {
    indices_4d = indices_4d.to(at::kInt);
  }

  auto grad_output_contiguous =
      cnnl_contiguous(gradOutput_4d, at::MemoryFormat::ChannelsLast);
  auto grad_input_contiguous =
      cnnl_contiguous(gradInput, at::MemoryFormat::ChannelsLast);
  auto input_contiguous =
      cnnl_contiguous(input_4d, at::MemoryFormat::ChannelsLast);
  auto indices_contiguous =
      cnnl_contiguous(indices_4d, at::MemoryFormat::ChannelsLast);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "adaptive_max_pool2d_backward_out_mlu",
      [&] {
        cnnl_adaptive_max_pool2d_backward_internal(
            grad_input_contiguous,
            grad_output_contiguous,
            input_contiguous,
            indices_contiguous);

        if (is_copy_necessary(gradInput, grad_input_contiguous)) {
          gradInput.copy_(grad_input_contiguous);
        }

        if (input.ndimension() == 3) {
          gradInput.squeeze_(0);
        }
      });
  return gradInput;
}

} // namespace ops
} // namespace torch_mlu
