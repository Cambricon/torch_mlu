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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_repeat(const at::Tensor& self, at::IntArrayRef repeats) {
  TORCH_CHECK(
      repeats.size() >= (size_t)self.dim(),
      "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
  // Add new leading dimensions to the tensor if the
  // number of target dimensions is larger than the
  // number of source dimensions.
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> target_size(repeats.size());
  bool zero_tensor = false;
  for (const auto idx : c10::irange(repeats.size())) {
    if (repeats[idx] == 0) {
      zero_tensor = true;
    }
    target_size[idx] = padded_size[idx] * repeats[idx];
  }

  // use contiguous memory format when input size is different with output size
  auto memory_format = num_new_dimensions > 0 ? c10::MemoryFormat::Contiguous
                                              : self.suggest_memory_format();
  auto input_contiguous = cnnl_contiguous(self, memory_format);
  auto output =
      at::empty(target_size, input_contiguous.options(), memory_format);

  // return an empty tensor if one of the repeat dimensions is zero
  if (zero_tensor) {
    return output;
  }

  cnnl_repeat_internal(output, input_contiguous);
  return output;
}

at::Tensor cnnl_repeat_backward(
    const at::Tensor& self,
    at::Tensor& grad,
    at::IntArrayRef repeats) {
  auto input_shape = self.sym_sizes();
  auto find_iter = std::find(repeats.cbegin(), repeats.cend(), 0);
  if (find_iter != repeats.cend()) {
    return at::zeros_symint(input_shape, grad.options());
  }
  const auto input_dims = input_shape.size();
  auto num_unsqueezed = grad.dim() - input_dims;
  for (const auto i : c10::irange(num_unsqueezed)) {
    (void)i; // Suppress unused variable warning
    grad = grad.sum(0, false);
  }

  // Origin algorithm will increase grad's dimensions by grad =
  // grad.reshape(grad_size), then do add operation on specific dimensions.
  // Reshape in this process is easily surpass MLU dimension limits. Currently
  // MLU uses an old algorithm from Pytorch1.6. Two algorithms have the same
  // functions but different implementation methods, and the performance of the
  // old algorithm is poor. For details:
  // https://github.com/pytorch/pytorch/issues/43192
  // https://github.com/pytorch/pytorch/pull/46726
  if (grad.device().type() == c10::DeviceType::PrivateUse1) {
    for (size_t j = num_unsqueezed; j < repeats.size(); ++j) {
      auto repeat = repeats[j];
      if (repeat == 1) {
        continue;
      }
      int64_t dim = j - num_unsqueezed;
      auto sum_tensorlist = [](at::TensorList tl) {
        if (tl.size() == 0) {
          throw std::runtime_error("Can't sum tensorlist of size 0");
        }
        at::Tensor sum = tl[0];
        for (size_t i = 1; i < tl.size(); ++i) {
          sum = sum + tl[i];
        }
        return sum;
      };

      grad = sum_tensorlist(grad.chunk(repeat, dim));
    }
    return grad;
  }
}

class RepeatFunction : public torch::autograd::Function<RepeatFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& self,
      const at::IntArrayRef repeats) {
    at::AutoDispatchBelowADInplaceOrView g;
    ctx->save_for_backward({self});
    ctx->saved_data["repeats"] = repeats;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("aten::repeat", "")
                         .typed<decltype(cnnl_repeat)>();
    auto result = op.call(self, repeats);
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    auto self = saved[0];
    std::vector<int64_t> repeats_vector =
        ctx->saved_data["repeats"].toIntVector();
    at::IntArrayRef repeats = repeats_vector;
    auto result = cnnl_repeat_backward(self, grad_output[0], repeats);
    return {result, at::Tensor()};
  }
};

at::Tensor cnnl_repeat_autograd(
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  auto result = RepeatFunction::apply(self, repeats);
  return result[0];
}

} // namespace ops
} // namespace torch_mlu
