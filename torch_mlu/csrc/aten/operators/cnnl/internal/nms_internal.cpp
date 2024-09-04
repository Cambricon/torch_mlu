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

at::Tensor cnnl_nms_internal(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {
  if (dets.numel() == 0)
    return at::empty({0}, dets.options().dtype(at::kLong));

  const cnnlNmsBoxPointMode_t box_mode = CNNL_NMS_BOX_DIAGONAL;
  const cnnlNmsOutputMode_t output_mode = CNNL_NMS_OUTPUT_TARGET_INDICES;
  const cnnlNmsAlgo_t algo = CNNL_NMS_ALGO_EXCLUDE_BOUNDARY;
  const cnnlNmsMethodMode_t method_mode = CNNL_NMS_HARD_NMS;
  const float soft_nms_sigma = 0.0;
  const int max_output_size = (int)dets.size(0);
  const float confidence_threshold = 0.0;
  const float offset = 0.0;
  const int input_layout = 0;
  const bool pad_to_max_output_size = false;

  auto output = at::empty({max_output_size}, dets.options().dtype(at::kInt));
  auto dets_impl = getMluTensorImpl(dets);
  auto scores_impl = getMluTensorImpl(scores);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();

  // get cnnl descriptor
  CnnlTensorDescriptor dets_desc;
  CnnlTensorDescriptor scores_desc;
  CnnlTensorDescriptor output_desc;
  dets_desc.set(dets);
  scores_desc.set(scores);
  output_desc.set(output);

  auto dets_ptr = dets_impl->mlu_data_ptr();
  auto scores_ptr = scores_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // set nms descriptor
  CnnlNmsDescriptor nms_desc;
  nms_desc.set(
      box_mode,
      output_mode,
      algo,
      method_mode,
      iou_threshold,
      soft_nms_sigma,
      max_output_size,
      confidence_threshold,
      offset,
      input_layout,
      pad_to_max_output_size);

  // prepare scores for nms3D
  auto scores_cnnl_desc = scores_desc.desc();
  if (scores.dim() == 0) {
    scores_ptr = nullptr;
    scores_cnnl_desc = nullptr;
  }

  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetNmsWorkspaceSize_v3(
      handle, dets_desc.desc(), scores_cnnl_desc, &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  auto output_size = at::empty({1}, scores.options().dtype(at::kInt));
  auto output_size_impl = getMluTensorImpl(output_size);
  auto output_size_ptr = output_size_impl->mlu_data_ptr();

  // calculate
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(dets.scalar_type(), "MLU nms", [&] {
    TORCH_CNNL_CHECK(cnnlNms_v2(
        handle,
        nms_desc.desc(),
        dets_desc.desc(),
        dets_ptr,
        scores_cnnl_desc,
        scores_ptr,
        workspace_ptr.get(),
        space_size,
        output_desc.desc(),
        output_ptr,
        output_size_ptr));
  });
  int output_num = *static_cast<int*>(output_size.cpu().data_ptr());
  return output.slice(0, 0, output_num).to(at::kLong);
}

} // namespace ops
} // namespace torch_mlu
