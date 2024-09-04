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

#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_nll_loss_forward_internal(
    at::Tensor& output,
    at::Tensor& total_weight,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto input_size = self.sizes().vec();
  int64_t C = self.dim() == 1 ? input_size[0] : input_size[1];
  int64_t N = self.numel();
  if (C != 0)
    N /= C;
  auto target_cast = target;
  // TODO(PYTORCH-9442): remove this when cnnl support uint8 target.
  if (target.scalar_type() != at::ScalarType::Int &&
      target.scalar_type() != at::ScalarType::Long) {
    target_cast = target.to(at::ScalarType::Int);
  }
  auto self_impl = getMluTensorImpl(self);
  auto target_impl = getMluTensorImpl(target_cast);
  cnnlNlllossAlgorithm_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_REDUCTION_NONE;
      break;
    case 1:
      reduction_mode = CNNL_REDUCTION_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_REDUCTION_SUM;
      break;
    default:
      LOG(ERROR) << "nll_loss reduciton mode is avaliable";
      break;
  }
  // get current handle
  auto handle = getCurrentHandle();

  // CNNL kernel can't handle 0 element, so hard code in here.
  if (self.numel() == 0) {
    total_weight.zero_();
    if (reduction_mode == CNNL_REDUCTION_SUM) {
      output.zero_();
    }
    if (reduction_mode == CNNL_REDUCTION_MEAN) {
      output.fill_(NAN);
    }
    return;
  }

  auto output_impl = getMluTensorImpl(output);
  auto tw_impl = getMluTensorImpl(total_weight);

  // get cnnl descriptor
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor target_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor tw_desc;
  CnnlTensorDescriptor output_desc;
  std::vector<int64_t> cnnl_input_size({N, C});
  std::vector<int64_t> input_stride({C, 1});
  std::vector<int64_t> target_size({N});
  std::vector<int64_t> weight_size({C});
  std::vector<int64_t> weight_stride({1});
  self_desc.set(self, cnnl_input_size, input_stride, CNNL_LAYOUT_ARRAY);
  target_desc.set(target_cast, target_size, weight_stride, CNNL_LAYOUT_ARRAY);

  void* weight_ptr = nullptr;
  if (weight.defined()) {
    auto weight_contiguous = cnnl_contiguous(weight);
    auto weight_impl = getMluTensorImpl(weight_contiguous);
    weight_ptr = mlu_data_ptr(weight_impl);
    weight_desc.set(
        weight_contiguous, weight_size, weight_stride, CNNL_LAYOUT_ARRAY);
  }
  reduction == 0
      ? output_desc.set(output, target_size, weight_stride, CNNL_LAYOUT_ARRAY)
      : output_desc.set(output);
  tw_desc.set(total_weight);

  // malloc mlu memory ( malloc and memcpy only really happen in the first time)
  auto self_ptr = mlu_data_ptr(self_impl);
  auto target_ptr = mlu_data_ptr(target_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto tw_ptr = mlu_data_ptr(tw_impl);

  // prepare workspace
  size_t sz = 0;
  TORCH_CNNL_CHECK(cnnlGetNlllossWorkspaceSize(handle, self_desc.desc(), &sz));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(sz);

  // calculate
  TORCH_CNNL_CHECK(cnnlNlllossForward_v2(
      handle,
      reduction_mode,
      self_desc.desc(),
      self_ptr,
      target_desc.desc(),
      target_ptr,
      ignore_index,
      weight.defined() ? weight_desc.desc() : nullptr,
      weight_ptr,
      ws_ptr.get(),
      sz,
      reduction != 0 ? tw_desc.desc() : nullptr,
      reduction != 0 ? tw_ptr : nullptr,
      output_desc.desc(),
      output_ptr));
}

void cnnl_nll_loss_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight) {
  auto input_size = self.sizes().vec();
  int64_t C = self.dim() == 1 ? input_size[0] : input_size[1];
  int64_t N = self.numel();
  if (C != 0)
    N /= C;
  auto target_cast = target;
  // TODO(PYTORCH-9442): remove this when cnnl support uint8 target.
  if (target.scalar_type() != at::ScalarType::Int &&
      target.scalar_type() != at::ScalarType::Long) {
    target_cast = target.to(at::ScalarType::Int);
  }
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto target_impl = getMluTensorImpl(target_cast);
  auto tw_impl = getMluTensorImpl(total_weight);
  cnnlNlllossAlgorithm_t reduction_mode;
  switch (reduction) {
    case 0:
      reduction_mode = CNNL_REDUCTION_NONE;
      break;
    case 1:
      reduction_mode = CNNL_REDUCTION_MEAN;
      break;
    case 2:
      reduction_mode = CNNL_REDUCTION_SUM;
      break;
    default:
      LOG(ERROR) << "nll_loss reduciton mode is avaliable";
      break;
  }
  // get current handle
  auto handle = getCurrentHandle();

  // deal with zero element tensor
  if (N == 0 || C == 0) {
    return;
  }
  auto grad_input_impl = getMluTensorImpl(grad_input);

  // get cnnl descriptor
  CnnlTensorDescriptor grad_output_desc;
  CnnlTensorDescriptor target_desc;
  CnnlTensorDescriptor weight_desc;
  CnnlTensorDescriptor tw_desc;
  CnnlTensorDescriptor grad_input_desc;
  std::vector<int64_t> target_size({N});
  std::vector<int64_t> grad_input_size({N, C});
  std::vector<int64_t> grad_input_stride({C, 1});
  std::vector<int64_t> weight_size({C});
  std::vector<int64_t> weight_stride({1});
  reduction == 0
      ? grad_output_desc.set(
            grad_output, target_size, weight_stride, CNNL_LAYOUT_ARRAY)
      : grad_output_desc.set(grad_output);
  target_desc.set(target_cast, target_size, weight_stride, CNNL_LAYOUT_ARRAY);
  void* weight_ptr = nullptr;
  if (weight.defined()) {
    auto weight_contiguous = cnnl_contiguous(weight);
    auto weight_impl = getMluTensorImpl(weight_contiguous);
    weight_ptr = mlu_data_ptr(weight_impl);
    weight_desc.set(
        weight_contiguous, weight_size, weight_stride, CNNL_LAYOUT_ARRAY);
  }
  grad_input_desc.set(
      grad_input, grad_input_size, grad_input_stride, CNNL_LAYOUT_ARRAY);
  tw_desc.set(total_weight);

  // malloc mlu memory ( malloc and memcpy only really happen in the first time)
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);
  auto target_ptr = mlu_data_ptr(target_impl);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);
  auto tw_ptr = mlu_data_ptr(tw_impl);
  // calculate
  TORCH_CNNL_CHECK(cnnlNlllossBackward_v2(
      handle,
      reduction_mode,
      grad_output_desc.desc(),
      grad_output_ptr,
      target_desc.desc(),
      target_ptr,
      ignore_index,
      weight.defined() ? weight_desc.desc() : nullptr,
      weight_ptr,
      reduction != 0 ? tw_desc.desc() : nullptr,
      reduction != 0 ? tw_ptr : nullptr,
      grad_input_desc.desc(),
      grad_input_ptr));
  return;
}

} // namespace ops
} // namespace torch_mlu
