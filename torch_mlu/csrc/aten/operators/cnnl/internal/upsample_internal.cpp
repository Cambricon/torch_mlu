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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_upsample_internal(
    const at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    bool align_center,
    cnnlInterpMode_t interp_mode,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    std::optional<double> scales_d,
    bool is_exact /* only used for nearest-exact */) {
  cnnlTensorLayout_t layout;
  if (self.dim() == 3) {
    layout = CNNL_LAYOUT_NLC;
  } else if (self.dim() == 4) {
    layout = CNNL_LAYOUT_NHWC;
  } else if (self.dim() == 5) {
    layout = CNNL_LAYOUT_NDHWC;
  } else {
    TORCH_CHECK(false, "unsupported self dim");
  }

  // malloc mlu memory
  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  auto descInput = getTensorDesc(input_impl, layout);
  auto descOutput = getTensorDesc(output_impl, layout);

  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  c10::SmallVector<float, 4> scales;
  if (scales_d.has_value()) {
    scales.push_back(scales_d.value());
  }
  if (scales_h.has_value()) {
    scales.push_back(scales_h.value());
  }
  if (scales_w.has_value()) {
    scales.push_back(scales_w.value());
  }
  auto* scales_ptr = scales.size() ? scales.data() : nullptr;

  CnnlInterpDescriptor descInterp;
  descInterp.set(
      descInput.get(),
      interp_mode,
      align_corners,
      align_center,
      scales_ptr,
      is_exact);

  auto handle = getCurrentHandle();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "MLU upsample",
      [&] {
        TORCH_CNNL_CHECK(cnnlInterp_v3(
            handle,
            descInterp.desc(),
            descInput.get(),
            input_ptr,
            descOutput.get(),
            output_ptr));
      });
}

void cnnl_upsample_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    bool align_corners,
    bool align_center,
    cnnlInterpMode_t interp_mode,
    std::optional<double> scales_h,
    std::optional<double> scales_w,
    std::optional<double> scales_d) {
  cnnlTensorLayout_t layout;
  if (grad_input.dim() == 3) {
    layout = CNNL_LAYOUT_NLC;
  } else if (grad_input.dim() == 4) {
    layout = CNNL_LAYOUT_NHWC;
  } else if (grad_input.dim() == 5) {
    layout = CNNL_LAYOUT_NDHWC;
  } else {
    TORCH_CHECK(false, "unsupported grad_input dim");
  }

  // malloc mlu memory
  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_output_impl = getMluTensorImpl(grad_output);

  auto descGradInput = getTensorDesc(grad_input_impl, layout);
  auto descGradOutput = getTensorDesc(grad_output_impl, layout);

  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);

  CnnlInterpBackwardDescriptor descInterp;
  cnnlInterpAlgo_t interp_algo;
  if (!align_corners && !align_center) {
    interp_algo = CNNL_INTERP_ALGO_0;
  } else if (!align_corners && align_center) {
    interp_algo = CNNL_INTERP_ALGO_1;
  } else if (align_corners && !align_center) {
    interp_algo = CNNL_INTERP_ALGO_2;
  } else if (align_corners && align_center) {
    interp_algo = CNNL_INTERP_ALGO_3;
  }

  c10::SmallVector<double, 4> scales;
  if (scales_d.has_value()) {
    scales.push_back(scales_d.value());
  }
  if (scales_h.has_value()) {
    scales.push_back(scales_h.value());
  }
  if (scales_w.has_value()) {
    scales.push_back(scales_w.value());
  }

  auto* scales_ptr = scales.size() ? scales.data() : nullptr;
  descInterp.set(descGradOutput.get(), interp_mode, interp_algo, scales_ptr);

  auto handle = getCurrentHandle();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad_output.scalar_type(),
      "MLU upsample backward",
      [&] {
        TORCH_CNNL_CHECK(cnnlInterpBackward_v3(
            handle,
            descInterp.desc(),
            descGradOutput.get(),
            grad_output_ptr,
            descGradInput.get(),
            grad_input_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
