
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
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {

TORCH_IMPL_FUNC(nll_loss_forward_out_mlu)
(const Tensor& self,
 const Tensor& target,
 const at::OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  int64_t n_dims = self.dim();
  int64_t batch_size = n_dims == 1 ? 1 : self.size(0);
  if (reduction == at::Reduction::None && self.dim() == 2) {
    at::native::resize_output(output, {batch_size});
    total_weight.zero_();
    if (batch_size == 0) {
      // This guards from unnecessary operations and launching kernel with 0
      // blocks.
      return;
    }
  } else {
    // produce scalar outputs for the reduction case
    at::native::resize_output(output, {});
    if (target.numel() == 0) {
      // Here target (and input) have zero elements
      // Mean reduction on empty tensors produces NaN. See the discussion in
      // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
      if (reduction == at::Reduction::Mean) {
        output.fill_(std::numeric_limits<double>::quiet_NaN());
      } else {
        output.zero_();
      }
      total_weight.zero_();
      return;
    }
  }
  const Tensor& weight = weight_opt.getTensorRef();

  auto self_contiguous = cnnl_contiguous(self);
  auto target_contiguous = cnnl_contiguous(target);
  auto output_contiguous = cnnl_contiguous(output);
  auto total_weight_contiguous = cnnl_contiguous(total_weight);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "nll_loss_forward_out_mlu",
      [&]() {
        AT_DISPATCH_INDEX_TYPES(
            target.scalar_type(), "nll_loss_forward_out_mlu_kernel_index", [&] {
              cnnl_nll_loss_forward_internal(
                  output_contiguous,
                  total_weight_contiguous,
                  self_contiguous,
                  target_contiguous,
                  weight,
                  reduction,
                  ignore_index);

              if (is_copy_necessary(output, output_contiguous)) {
                output.copy_(output_contiguous);
              }

              if (is_copy_necessary(total_weight, total_weight_contiguous)) {
                total_weight.copy_(total_weight_contiguous);
              }
            });
      });
}

// input tensor should be 1D or 2D
TORCH_IMPL_FUNC(nll_loss_backward_out_mlu)
(const Tensor& grad_output,
 const Tensor& self,
 const Tensor& target,
 at::OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input) {
  const Tensor& weight = weight_opt.getTensorRef();
  grad_input.zero_();
  int64_t n_dims = self.dim();
  int64_t batch_size = n_dims == 1 ? 1 : self.size(0);
  if (batch_size == 0) {
    // This guards from unnecessary operations and launching kernel with 0
    // blocks.
    return;
  }

  auto grad_output_contiguous = cnnl_contiguous(grad_output);
  auto self_contiguous = cnnl_contiguous(self);
  auto target_contiguous = cnnl_contiguous(target);
  auto total_weight_contiguous = cnnl_contiguous(total_weight);
  auto grad_input_contiguous = cnnl_contiguous(grad_input);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "nll_loss_backward_out_mlu",
      [&]() {
        AT_DISPATCH_INDEX_TYPES(
            target.scalar_type(),
            "nll_loss_backward_out_mlu_kernel_index",
            [&] {
              cnnl_nll_loss_backward_internal(
                  grad_input_contiguous,
                  grad_output_contiguous,
                  self_contiguous,
                  target_contiguous,
                  weight,
                  reduction,
                  ignore_index,
                  total_weight_contiguous);

              if (is_copy_necessary(grad_input, grad_input_contiguous)) {
                grad_input.copy_(grad_input_contiguous);
              }
            });
      });
}

void check_inputs_nll_loss2d(
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight) {
  TORCH_CHECK(
      target.dim() == 3,
      "only batches of spatial targets supported (3D tensors)"
      " but got targets of size: : ",
      target.sizes());
  TORCH_CHECK(
      input.dim() == 4,
      "only batches of spatial inputs supported (4D tensors), "
      "but got input of size: ",
      input.sizes());
  TORCH_CHECK(
      !weight.defined() || weight.numel() == input.size(1),
      "weight tensor should be defined either for all or no classes");

  TORCH_CHECK(
      input.size(0) == target.size(0) && input.size(2) == target.size(1) &&
          input.size(3) == target.size(2),
      "input and target batch or spatial sizes don't match: target ",
      target.sizes(),
      ", input ",
      input.sizes());
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_nll_loss2d_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& output,
    at::Tensor& total_weight) {
  /*
   * transform nll_loss2d to nll_loss
   * ==> nll_loss2d
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, N, d1, d2]  (nll_loss2d)  [m, d1, d2, N]   {input}
   * [m, d1, d2]                   [m, d2, d1]      {target}
   *
   *      ||                            ||
   *
   * [mxd1xd2, N]                  [mxd1xd2, N]
   * [mxd1xd2]                     [mxd1xd2]
   *
   * ==> nll_loss
   */
  const Tensor& weight = *at::borrow_from_optional_tensor(weight_opt);
  check_inputs_nll_loss2d(self, target, weight);
  total_weight.resize_({});
  int64_t batch_size = self.size(0);
  int64_t H = self.size(2);
  int64_t W = self.size(3);
  int64_t count = batch_size * H * W;
  if (reduction == at::Reduction::None) {
    at::native::resize_output(output, {batch_size, H, W});
    if (count == 0) {
      return std::tuple<Tensor&, Tensor&>(output, total_weight);
    }
  } else {
    // produce scalar outputs for the reduction case
    at::native::resize_output(output, {});
    if (target.numel() == 0) {
      // Here target (and input) have zero elements
      // Mean reduction on empty tensors produces NaN. See the discussion in
      // https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
      if (reduction == at::Reduction::Mean) {
        output.fill_(std::numeric_limits<double>::quiet_NaN());
      } else {
        output.zero_();
      }
      total_weight.zero_();
      return std::tuple<Tensor&, Tensor&>(output, total_weight);
    }
  }

  /*
   * transform self
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, N, d1, d2]  (nll_loss2d)  [m, d1, d2, N]   {input}
   *
   * [mxd1xd2, N]                  [mxd1xd2, N]
   * [mxd1xd2]                     [mxd1xd2]
   */

  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);

  /*
   * transform weight
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, d1, d2]                   [m, d2, d1]      {target}
   *
   * [m, d1, d2]  (channels_last)  [m, d1, d2]
   *
   * [mxd1xd2]                     [mxd1xd2]
   *
   */
  auto target_contiguous =
      cnnl_contiguous(target, target.suggest_memory_format());
  auto output_contiguous = cnnl_contiguous(output);
  auto total_weight_contiguous = cnnl_contiguous(total_weight);

  output.zero_();
  total_weight.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Half,
      self.scalar_type(),
      "cnnl_nll_loss2d_forward_out",
      [&]() {
        // nll_loss
        cnnl_nll_loss_forward_internal(
            output_contiguous,
            total_weight_contiguous,
            self_contiguous,
            target_contiguous,
            weight,
            reduction,
            ignore_index);
        if (is_copy_necessary(output, output_contiguous)) {
          output.copy_(output_contiguous);
        }

        if (is_copy_necessary(total_weight, total_weight_contiguous)) {
          total_weight.copy_(total_weight_contiguous);
        }
      });
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<at::Tensor, at::Tensor> cnnl_nll_loss2d_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  cnnl_nll_loss2d_forward_out(
      self, target, weight, reduction, ignore_index, output, total_weight);
  return std::make_tuple(output, total_weight);
}

at::Tensor& cnnl_nll_loss2d_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input) {
  /*
   * transform nll_loss2d bp to nll_loss bp
   * ==> nll_loss2d
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, N, d1, d2]  (nll_loss2d)  [m, d1, d2, N]   {input}
   * [m, d1, d2]                   [m, d2, d1]      {target}
   *
   *      ||                            ||
   *
   * [mxd1xd2, N]                  [mxd1xd2, N]
   * [mxd1xd2]                     [mxd1xd2]
   *
   * ==> nll_loss
   */
  const Tensor& weight = *at::borrow_from_optional_tensor(weight_opt);
  check_inputs_nll_loss2d(self, target, weight);
  grad_input.resize_as_(self);
  grad_input.zero_();
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");
  if (reduction == at::Reduction::None) {
    TORCH_CHECK(
        grad_output.dim() == 3,
        "grad_output must have same dimension as target (3) but got dimension: ",
        grad_output.sizes());
    TORCH_CHECK(
        grad_output.size(0) == target.size(0) &&
            grad_output.size(1) == target.size(1) &&
            grad_output.size(2) == target.size(2),
        "grad_output sizes don't match target sizes: target ",
        target.sizes(),
        ", grad_output ",
        grad_output.sizes())
    int64_t batch_size = self.size(0);
    int64_t H = self.size(2);
    int64_t W = self.size(3);
    int64_t count = batch_size * H * W;

    if (count == 0) {
      return grad_input;
    }
  }

  /*
   * transform self
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, N, d1, d2]  (nll_loss2d)  [m, d1, d2, N]   {input}
   *
   * [mxd1xd2, N]                  [mxd1xd2, N]
   * [mxd1xd2]                     [mxd1xd2]
   */

  auto memory_format = get_channels_last_memory_format(self.dim());
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto grad_input_contiguous = cnnl_contiguous(grad_input, memory_format);

  /*
   * transform weight
   *
   *    CPU(NCHW)        <==>         MLU(NHWC)
   * [m, d1, d2]                   [m, d2, d1]      {target}
   *
   * [m, d1, d2]  (channels_last)  [m, d1, d2]
   *
   * [mxd1xd2]                     [mxd1xd2]
   *
   */
  auto target_contiguous =
      cnnl_contiguous(target, target.suggest_memory_format());
  auto total_weight_contiguous = cnnl_contiguous(total_weight);
  auto grad_output_contiguous = cnnl_contiguous(grad_output);

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::Half,
      self.scalar_type(),
      "cnnl_nll_loss2d_backward_out",
      [&]() {
        // nll_loss bakcward
        cnnl_nll_loss_backward_internal(
            grad_input_contiguous,
            grad_output_contiguous,
            self_contiguous,
            target_contiguous,
            weight,
            reduction,
            ignore_index,
            total_weight_contiguous);

        if (is_copy_necessary(grad_input, grad_input_contiguous)) {
          grad_input.copy_(grad_input_contiguous);
        }
      });
  return grad_input;
}

at::Tensor cnnl_nll_loss2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  auto grad_input = at::empty_like(self);
  cnnl_nll_loss2d_backward_out(
      grad_output,
      self,
      target,
      weight,
      reduction,
      ignore_index,
      total_weight,
      grad_input);
  return grad_input;
}

} // namespace ops
} // namespace torch_mlu
