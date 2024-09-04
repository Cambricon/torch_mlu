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

void cnnl_smooth_l1_loss_forward_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  auto input_impl = getMluTensorImpl(input);
  auto input_desc = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto target_impl = getMluTensorImpl(target);
  auto target_desc = getTensorDesc(target_impl);
  auto target_ptr = mlu_data_ptr(target_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // prepare reduction_mode
  cnnlSmoothL1LossAlgorithm_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_NONE;
      break;
    case 1:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_SUM;
      break;
    default:
      TORCH_CHECK(false, "unsupported reduction mode");
      break;
  }

  // get current handle
  auto handle = getCurrentHandle();

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetSmoothL1LossForwardWorkspaceSize(
      handle, input_desc.get(), reduction_mode, &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // calculate
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "MLU smooth_l1_loss", [&] {
        TORCH_CNNL_CHECK(cnnlSmoothL1LossForward_v2(
            handle,
            input_desc.get(),
            input_ptr,
            target_desc.get(),
            target_ptr,
            beta,
            reduction_mode,
            workspace_ptr.get(),
            workspace_size,
            output_desc.get(),
            output_ptr));
      });
}

void cnnl_smooth_l1_loss_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double beta) {
  auto input_impl = getMluTensorImpl(input);
  auto input_desc = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);
  tensorDescPtr_t grad_output_desc;
  if (reduction == 1 || reduction == 2) {
    // grad_output's shape is [1,1,1,1] after TensorIterator
    // and set [1] for CNNL size
    const int64_t val = 1;
    at::IntArrayRef shape_and_stride(val);
    cnnlDataType_t data_type = getCnnlType(grad_output_impl);
    cnnlTensorLayout_t layout = suggestCnnlLayout(grad_output_impl);
    grad_output_desc =
        getTensorDesc(shape_and_stride, shape_and_stride, data_type, layout);
  } else {
    grad_output_desc = getTensorDesc(grad_output_impl);
  }

  auto target_impl = getMluTensorImpl(target);
  auto target_desc = getTensorDesc(target_impl);
  auto target_ptr = mlu_data_ptr(target_impl);

  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto grad_input_desc = getTensorDesc(grad_input_impl);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);

  cnnlSmoothL1LossAlgorithm_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_NONE;
      break;
    case 1:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_SMOOTHL1LOSS_REDUCTION_SUM;
      break;
    default:
      TORCH_CHECK(false, "unsupported reduction mode");
      break;
  }

  // get current handle
  auto handle = getCurrentHandle();

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetSmoothL1LossBackwardWorkspaceSize(
      handle, input_desc.get(), reduction_mode, &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // calculate
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "MLU smooth_l1_loss_backward", [&] {
        TORCH_CNNL_CHECK(cnnlSmoothL1LossBackward_v2(
            handle,
            input_desc.get(),
            input_ptr,
            target_desc.get(),
            target_ptr,
            grad_output_desc.get(),
            grad_output_ptr,
            beta,
            reduction_mode,
            workspace_ptr.get(),
            workspace_size,
            grad_input_desc.get(),
            grad_input_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
