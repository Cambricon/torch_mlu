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

inline cnnlInterpMode_t get_interp_mode(int64_t interpolation_mode) {
  cnnlInterpMode_t interp_mode;
  switch (interpolation_mode) {
    case 0:
      interp_mode = CNNL_INTERP_BILINEAR;
      break;
    case 1:
      interp_mode = CNNL_INTERP_NEAREST;
      break;
    case 2:
      interp_mode = CNNL_INTERP_BICUBIC;
      break;
    default:
      AT_ERROR(
          "grid sample only support nearest, bilinear and bicubic interp_mode on MLU.");
  }
  return interp_mode;
}

inline cnnlGridSamplePaddingMode_t get_pad_mode(int64_t padding_mode) {
  cnnlGridSamplePaddingMode_t pad_mode;
  switch (padding_mode) {
    case 0:
      pad_mode = CNNL_GRIDSAMPLE_PADDING_ZEROS;
      break;
    case 1:
      pad_mode = CNNL_GRIDSAMPLE_PADDING_BORDER;
      break;
    case 2:
      pad_mode = CNNL_GRIDSAMPLE_PADDING_REFLECTION;
      break;
    default:
      AT_ERROR(
          "grid sample only support zero, border and reflection padding mode on MLU.");
  }
  return pad_mode;
}

at::Tensor& cnnl_grid_sampler_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool is_3d,
    bool align_corners) {
  cnnlInterpMode_t interp_mode = get_interp_mode(interpolation_mode);
  cnnlGridSamplePaddingMode_t pad_mode = get_pad_mode(padding_mode);
  CnnlGridSampleDescriptor op_desc;
  op_desc.set(interp_mode, pad_mode, align_corners);

  cnnlTensorLayout_t layout = is_3d ? CNNL_LAYOUT_NDHWC: CNNL_LAYOUT_NHWC;

  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->mlu_data_ptr();
  CnnlTensorDescriptor input_desc;
  input_desc.set(input, layout);

  auto grid_impl = getMluTensorImpl(grid);
  auto grid_ptr = grid_impl->mlu_data_ptr();
  CnnlTensorDescriptor grid_desc;
  grid_desc.set(
      grid,
      grid.sizes().vec(),
      get_contiguous_strides(grid.sizes().vec()),
      layout);
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();
  CnnlTensorDescriptor output_desc;
  output_desc.set(output, layout);

  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetGridSampleForwardWorkspaceSize(
      handle,
      input_desc.desc(),
      grid_desc.desc(),
      output_desc.desc(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlGridSampleForward(
      handle,
      op_desc.desc(),
      input_desc.desc(),
      input_ptr,
      grid_desc.desc(),
      grid_ptr,
      output_desc.desc(),
      output_ptr,
      workspace_ptr.get(),
      workspace_size));
  return output;
}

void cnnl_grid_sampler_backward_internal(
    at::Tensor& grad_input,
    at::Tensor& grad_grid,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool is_3d,
    bool align_corners) {
  cnnlInterpMode_t interp_mode = get_interp_mode(interpolation_mode);
  cnnlGridSamplePaddingMode_t pad_mode = get_pad_mode(padding_mode);
  CnnlGridSampleDescriptor op_desc;
  op_desc.set(interp_mode, pad_mode, align_corners);
  
  cnnlTensorLayout_t layout = is_3d ? CNNL_LAYOUT_NDHWC: CNNL_LAYOUT_NHWC;

  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_output_ptr = grad_output_impl->mlu_data_ptr();
  CnnlTensorDescriptor grad_output_desc;
  grad_output_desc.set(grad_output, layout);

  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->mlu_data_ptr();
  CnnlTensorDescriptor input_desc;
  input_desc.set(input, layout);

  auto grid_impl = getMluTensorImpl(grid);
  auto grid_ptr = grid_impl->mlu_data_ptr();
  CnnlTensorDescriptor grid_desc;
  grid_desc.set(
      grid,
      grid.sizes().vec(),
      get_contiguous_strides(grid.sizes().vec()),
      layout);

  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_input_ptr = grad_input_impl->mlu_data_ptr();
  CnnlTensorDescriptor grad_input_desc;
  grad_input_desc.set(grad_input, layout);

  auto grad_grid_impl = getMluTensorImpl(grad_grid);
  auto grad_grid_ptr = grad_grid_impl->mlu_data_ptr();
  CnnlTensorDescriptor grad_grid_desc;
  grad_grid_desc.set(
      grad_grid,
      grad_grid.sizes().vec(),
      get_contiguous_strides(grad_grid.sizes().vec()),
      layout);

  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetGridSampleBackwardWorkspaceSize(
      handle,
      grad_output_desc.desc(),
      input_desc.desc(),
      grid_desc.desc(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlGridSampleBackward(
      handle,
      op_desc.desc(),
      grad_output_desc.desc(),
      grad_output_ptr,
      input_desc.desc(),
      input_ptr,
      grid_desc.desc(),
      grid_ptr,
      workspace_ptr.get(),
      workspace_size,
      grad_input_desc.desc(),
      grad_input_ptr,
      grad_grid_desc.desc(),
      grad_grid_ptr));

  return;
}

} // namespace ops
} // namespace torch_mlu
