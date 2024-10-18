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
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void check_grid_sampler_inputs(
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode) {
  TORCH_CHECK(
      input.dim() == 4 || input.dim() == 5,
      "grid_sampler(): expected 4D input or 5D input "
      "but got input with dim ",
      input.dim());
  TORCH_CHECK(
      input.dim() == grid.dim(),
      "grid_sampler(): expected 4D/5D input and grid with same number of "
      "dimensions, but got input with sizes ",
      input.sizes(),
      " and grid with sizes ",
      grid.sizes());
  TORCH_CHECK(
      input.dtype() == grid.dtype(),
      "grid_sampler(): expected input and grid to have same dtype, but "
      "got input with dtype ",
      input.dtype(),
      " and grid with dtype ",
      grid.dtype());
  // TODO(tdai): interpolation_mode: only support bilinear and nearest mode,
  // bicubic is not supported now. padding_mode: nearest only support zeros
  // mode, bilinear only support zeros and reflection.
  if (input.dim() == 5) {
    TORCH_CHECK(
        padding_mode == 0 && interpolation_mode == 0,
        "grid sampler 3d only support bilinear mode and zeros padding.");
  }
  TORCH_CHECK(
      interpolation_mode == 0 || interpolation_mode == 1,
      "interpolation_mode only support bilinear or nearest.");
  if (interpolation_mode == 0) {
    TORCH_MLU_CHECK(
        padding_mode == 0 || padding_mode == 2,
        "bilinear only support zeros or reflection padding.");
  }
  if (interpolation_mode == 1) {
    TORCH_MLU_CHECK(padding_mode == 0, "nearest only support zeros padding.");
  }
}

at::Tensor cnnl_grid_sampler_2d(
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  check_grid_sampler_inputs(input, grid, interpolation_mode, padding_mode);

  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = N * H * W;
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  // (N, H, W, 2)
  auto grid_contiguous = cnnl_contiguous(grid);
  auto output = at::empty(
      {N, input_contiguous.size(1), H, W},
      input_contiguous.options(),
      memory_format);
  if (count > 0) {
    return cnnl_grid_sampler_internal(
        output,
        input_contiguous,
        grid_contiguous,
        interpolation_mode,
        padding_mode,
	false,
        align_corners);
  }
  return output;
}

at::Tensor cnnl_grid_sampler_3d(
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners) {
  check_grid_sampler_inputs(input, grid, interpolation_mode, padding_mode);

  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  int64_t count = N * D * H * W;

  // cnnl Layout NDHWC
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto C = input_contiguous.size(1);
  auto grid_contiguous = cnnl_contiguous(grid);
  auto output = at::empty(
      {N, C, D, H, W},
      input_contiguous.options(),
      memory_format);
  if (count > 0) {
    return cnnl_grid_sampler_internal(
        output,
        input_contiguous,
        grid_contiguous,
        interpolation_mode,
        padding_mode,
	true,
        align_corners);
  }
  return output;
}

std::tuple<at::Tensor, at::Tensor> cnnl_grid_sampler_2d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  check_grid_sampler_inputs(input, grid, interpolation_mode, padding_mode);
  TORCH_MLU_CHECK(grad_output.dim() == 4, "grad_output dim must be 4.");

  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  int64_t count = N * H * W;
  auto memory_format = get_channels_last_memory_format(input.dim());
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  // (N, H, W, 2)
  auto grid_contiguous = cnnl_contiguous(grid);
  auto grad_input_contiguous = at::zeros_like(
      input_contiguous, input_contiguous.options(), memory_format);
  auto grad_grid_contiguous =
      at::empty_like(grid_contiguous, grid_contiguous.options());

  if (count > 0) {
    cnnl_grid_sampler_backward_internal(
        grad_input_contiguous,
        grad_grid_contiguous,
        grad_output_contiguous,
        input_contiguous,
        grid_contiguous,
        interpolation_mode,
        padding_mode,
	false,
        align_corners);
  }

  auto input_requires_grad = output_mask[0];
  at::Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return grad_input_contiguous;
    } else {
      return Tensor();
    }
  })();

  return std::make_tuple(grad_input, grad_grid_contiguous);
}

std::tuple<at::Tensor, at::Tensor> cnnl_grid_sampler_3d_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask) {
  check_grid_sampler_inputs(input, grid, interpolation_mode, padding_mode);
  TORCH_CHECK(grad_output.dim() == 5, "grad_output dim must be 5.");

  auto N = input.size(0);
  auto H = grid.size(1);
  auto W = grid.size(2);
  auto C = grid.size(3);
  int64_t count = N * H * W * C;
  // cnnl Layout NDHWC
  auto memory_format = at::MemoryFormat::ChannelsLast3d;
  auto input_contiguous = cnnl_contiguous(input, memory_format);
  auto grad_output_contiguous = cnnl_contiguous(grad_output, memory_format);
  auto grid_contiguous = cnnl_contiguous(grid);
  auto grad_input_contiguous = at::zeros_like(
      input_contiguous, input_contiguous.options(), memory_format);
  auto grad_grid_contiguous =
      at::empty_like(grid_contiguous, grid_contiguous.options());
  if (count > 0) {
    cnnl_grid_sampler_backward_internal(
        grad_input_contiguous,
        grad_grid_contiguous,
        grad_output_contiguous,
        input_contiguous,
        grid_contiguous,
        interpolation_mode,
        padding_mode,
	true,
        align_corners);
  }

  auto input_requires_grad = output_mask[0];
  at::Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return grad_input_contiguous;
    } else {
      return Tensor();
    }
  })();

  return std::make_tuple(grad_input, grad_grid_contiguous);
}
} // namespace ops
} // namespace torch_mlu
