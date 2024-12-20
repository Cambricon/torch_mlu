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

#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& bang__amp_update_scale_(
    at::Tensor& self,
    at::Tensor& growth_tracker,
    const at::Tensor& found_inf,
    double scale_growth_factor,
    double scale_backoff_factor,
    int64_t growth_interval) {
  TORCH_MLU_CHECK(
      growth_tracker.device().is_privateuseone(),
      "growth_tracker must be a MLU tensor.");
  TORCH_MLU_CHECK(
      self.device().is_privateuseone(), "self must be a MLU tensor.");
  TORCH_MLU_CHECK(
      found_inf.device().is_privateuseone(), "found_inf must be a MLU tensor.");
  TORCH_MLU_CHECK(
      growth_tracker.numel() == 1,
      "growth_tracker must be a 1-element tensor.");
  TORCH_MLU_CHECK(self.numel() == 1, "self must be a 1-element tensor.");
  TORCH_MLU_CHECK(
      found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_MLU_CHECK(
      growth_tracker.scalar_type() == at::ScalarType::Int,
      "growth_tracker must be an int tensor.");
  TORCH_MLU_CHECK(
      self.scalar_type() == at::ScalarType::Float,
      "self must be a float tensor.");
  TORCH_MLU_CHECK(
      found_inf.scalar_type() == at::ScalarType::Float,
      "found_inf must be a float tensor.");

  // self ptr
  auto current_scale_memory_format = self.suggest_memory_format();
  auto current_scale_contiguous =
      cnnl_contiguous(self, current_scale_memory_format);

  auto new_scale = at::empty(
      current_scale_contiguous.sizes(),
      current_scale_contiguous.options(),
      current_scale_memory_format);
  auto current_scale_impl = getMluTensorImpl(current_scale_contiguous);
  auto new_scale_impl = getMluTensorImpl(new_scale);
  auto current_scale_ptr = current_scale_impl->mlu_data_ptr();
  auto new_scale_ptr = new_scale_impl->mlu_data_ptr();

  // growth_tracker ptr
  auto growth_tracker_memory_format = growth_tracker.suggest_memory_format();
  auto growth_tracker_contiguous =
      cnnl_contiguous(growth_tracker, growth_tracker_memory_format);
  auto growth_tracker_impl = getMluTensorImpl(growth_tracker_contiguous);
  auto growth_tracker_ptr = growth_tracker_impl->mlu_data_ptr();

  // found_inf ptr
  auto found_inf_memory_format = found_inf.suggest_memory_format();
  auto found_inf_contiguous =
      cnnl_contiguous(found_inf, found_inf_memory_format);
  auto found_inf_impl = getMluTensorImpl(found_inf_contiguous);
  auto found_inf_ptr = found_inf_impl->mlu_data_ptr();

  // amp unscale
  cnrtDim3_t dim = {1, 1, 1};
  cnrtFunctionType_t ktype = cnrtFuncTypeBlock;
  auto compute_stream = getCurrentMLUStream();

  amp_update_scale_internal(
      new_scale_ptr,
      growth_tracker_ptr,
      growth_tracker_ptr,
      current_scale_ptr,
      found_inf_ptr,
      static_cast<float>(scale_growth_factor),
      static_cast<float>(scale_backoff_factor),
      growth_interval,
      dim,
      ktype,
      compute_stream.stream());
  self.copy_(new_scale);
  return self;
}

} // namespace ops
} // namespace torch_mlu
