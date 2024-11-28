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

#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_div_out_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& other,
    const std::string& rounding_mode) {
  if (input.numel() == 0 || other.numel() == 0) {
    return output;
  }
  if (rounding_mode == "true") {
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()) &&
            at::isFloatingType(other.scalar_type()),
        "div inputs only support floating type");
  } else {
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()) ||
            at::isIntegralType(input.scalar_type()),
        "div trunc/floor inputs only support floating/integral type");
    TORCH_CHECK(
        at::isFloatingType(other.scalar_type()) ||
            at::isIntegralType(other.scalar_type()),
        "div trunc/floor inputs only support floating/integral type");
  }
  // get current handle
  auto handle = getCurrentHandle();
  auto input_impl = getMluTensorImpl(input);
  auto other_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(output);

  // get cnnl desc
  auto cnnl_suggest_layout = suggestCnnlLayout(output_impl);
  auto desc_input = getTensorDesc(input_impl, cnnl_suggest_layout);
  auto desc_other = getTensorDesc(other_impl, cnnl_suggest_layout);
  auto desc_output = getTensorDesc(output_impl, cnnl_suggest_layout);

  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto other_ptr = mlu_data_ptr(other_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  if (input_impl->numel() == 0)
    return output;

  // workspace
  size_t workspace_size = 0;
  if (rounding_mode == "true") {
    TORCH_CNNL_CHECK(cnnlGetDivWorkspaceSize(
        handle,
        desc_input.get(),
        desc_other.get(),
        desc_output.get(),
        &workspace_size));
  } else if (rounding_mode == "trunc") {
    TORCH_CNNL_CHECK(cnnlGetFloorDivTruncWorkspaceSize(
        handle,
        desc_input.get(),
        desc_other.get(),
        desc_output.get(),
        &workspace_size));
  } else if (rounding_mode == "floor") {
    TORCH_CNNL_CHECK(cnnlGetFloorDivWorkspaceSize(
        handle,
        desc_input.get(),
        desc_other.get(),
        desc_output.get(),
        &workspace_size));
  }
  auto temp_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // set descriptor config
  if (rounding_mode == "true") {
    TORCH_CNNL_CHECK(cnnlDiv(
        handle,
        desc_input.get(),
        input_ptr,
        desc_other.get(),
        other_ptr,
        temp_ptr.get(),
        workspace_size,
        desc_output.get(),
        output_ptr));
  } else if (rounding_mode == "trunc") {
    TORCH_CNNL_CHECK(cnnlFloorDivTrunc_v2(
        handle,
        desc_input.get(),
        input_ptr,
        desc_other.get(),
        other_ptr,
        desc_output.get(),
        output_ptr,
        temp_ptr.get(),
        workspace_size));
  } else if (rounding_mode == "floor") {
    // cnnl FloorDiv_v2 use CNNL_COMPUTATION_FAST mode will cause
    // performace go down, use CNNL_COMPUTATION_HIGH_PRECISION instead;
    cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
    TORCH_CNNL_CHECK(cnnlFloorDivV2(
        handle,
        prefer,
        desc_input.get(),
        input_ptr,
        desc_other.get(),
        other_ptr,
        desc_output.get(),
        output_ptr,
        temp_ptr.get(),
        workspace_size));
  }
  return output;
}

} // namespace ops
} // namespace torch_mlu
