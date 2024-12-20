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

#include "aten/utils/cnnl_util.h"
#include "aten/utils/utils.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"

namespace torch_mlu {
namespace ops {

void amp_unscale_impl(
    std::vector<void*>& tensors_ptr,
    const std::vector<uint64_t>& tensors_numel,
    void* found_inf_ptr,
    void* inv_scale_ptr,
    cnrtDataType_V2_t cnrt_type) {
  // get current stream
  auto compute_stream = getCurrentMLUStream();

  cnrtFunctionType_t func_type = cnrtFuncTypeUnion1;
  uint32_t union_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  cnrtDim3_t k_dim = {core_dim, union_number, 1};

  size_t tensors_num = tensors_ptr.size();
  if (tensors_num == 1) {
    TORCH_MLU_CHECK(
        amp_unscale_internal(
            tensors_ptr[0],
            found_inf_ptr,
            inv_scale_ptr,
            found_inf_ptr,
            (int32_t)tensors_numel[0],
            func_type,
            k_dim,
            compute_stream.stream(),
            cnrt_type),
        "amp_unscale_internal call failed.");
  } else {
    TORCH_MLU_CHECK(
        amp_unscale_internal(
            tensors_ptr.data(),
            tensors_numel.data(),
            found_inf_ptr,
            inv_scale_ptr,
            found_inf_ptr,
            tensors_num,
            func_type,
            k_dim,
            compute_stream.stream(),
            cnrt_type),
        "amp_unscale_internal call failed.");
  }
}

void bang__amp_foreach_non_finite_check_and_unscale_(
    at::TensorList scaled_grads,
    at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  size_t n_tensors = scaled_grads.size();
  if (n_tensors == 0)
    return;

  TORCH_MLU_CHECK(
      inv_scale.device().is_privateuseone(), "inv_scale must be a MLU tensor.");
  TORCH_MLU_CHECK(
      found_inf.device().is_privateuseone(), "found_inf must be a MLU tensor.");
  TORCH_MLU_CHECK(
      inv_scale.numel() == 1, "inv_scale must be a 1-element tensor.");
  TORCH_MLU_CHECK(
      found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_MLU_CHECK(
      inv_scale.scalar_type() == at::ScalarType::Float,
      "inv_scale must be a float tensor.");
  TORCH_MLU_CHECK(
      found_inf.scalar_type() == at::ScalarType::Float,
      "found_inf must be a float tensor.");
  TORCH_MLU_CHECK(n_tensors > 0, "Tensor list must have at least one tensor.");

  // found_inf impl ptr
  auto found_inf_ptr = getMluTensorImpl(found_inf)->mlu_data_ptr();
  // inv_scale ptr
  auto inv_scale_ptr = getMluTensorImpl(inv_scale)->mlu_data_ptr();

  std::vector<void*> tensors_ptr;
  std::vector<uint64_t> tensors_numel;
  std::vector<at::Tensor> contiguous_tensors;

  auto device = scaled_grads[0].device();
  auto scalar_type = scaled_grads[0].scalar_type();
  auto cnrt_type =
      cnnlType2CnrtType_V2(getCnnlType(getMluTensorImpl(scaled_grads[0])));
  TORCH_MLU_CHECK(
      cnrtFloat == cnrt_type || cnrtHalf == cnrt_type,
      "Currently amp_unscale only support float32 and float16 dtype, not implemented for ",
      toString(scalar_type));

  uint64_t total_numel = 0;
  for (size_t i = 0; i < n_tensors; i++) {
    auto memory_format = scaled_grads[i].suggest_memory_format();
    const auto& t = scaled_grads[i];
    TORCH_MLU_CHECK(
        t.device().is_privateuseone(),
        "one of scaled_grads was not a mlu tensor.");
    // the check whether the input tensor is on the same device has been done in
    // wrapper and is deleted here.
    TORCH_MLU_CHECK(
        t.layout() == at::kStrided,
        "one of scaled_grads was not a strided tensor.");
    TORCH_MLU_CHECK(
        t.scalar_type() == scalar_type,
        "scaled_grads must be on the same type.");
    uint64_t t_numel = t.numel();
    if (t_numel == 0) {
      contiguous_tensors.emplace_back(t);
      continue;
    }
    total_numel += t_numel;

    if (t_numel > INT32_MAX) {
      TORCH_MLU_CHECK(false, "Not implemented large tensor.");
    }

    if (total_numel > INT32_MAX &&
        Global::instance().getDeviceName() < MLU590) {
      amp_unscale_impl(
          tensors_ptr, tensors_numel, found_inf_ptr, inv_scale_ptr, cnrt_type);
      tensors_ptr.clear();
      tensors_numel.clear();
      total_numel = 0;
      total_numel += t_numel;
    }

    // tensor impl ptr
    auto c_tensor = cnnl_contiguous(t, memory_format);
    contiguous_tensors.emplace_back(c_tensor);
    tensors_ptr.emplace_back(getMluTensorImpl(c_tensor)->mlu_data_ptr());
    // tensors numel
    tensors_numel.emplace_back(t_numel);
  }

  if (total_numel == 0) {
    return;
  }

  amp_unscale_impl(
      tensors_ptr, tensors_numel, found_inf_ptr, inv_scale_ptr, cnrt_type);
  for (size_t i(0); i < n_tensors; i++) {
    if (is_copy_necessary(scaled_grads[i], contiguous_tensors[i])) {
      scaled_grads[i].copy_(contiguous_tensors[i]);
    }
  }
  return;
}

at::Tensor bang_amp_unscale(
    at::TensorList scaled_grads,
    const at::Tensor& found_inf,
    const at::Tensor& inv_scale) {
  TORCH_WARN_ONCE(
      "The torch.ops.torch_mlu.amp_unscale is deprecated and will be removed. ",
      "Use torch._amp_foreach_non_finite_check_and_unscale_ instead.");
  bang__amp_foreach_non_finite_check_and_unscale_(
      scaled_grads, const_cast<at::Tensor&>(found_inf), inv_scale);
  return found_inf;
}

} // namespace ops
} // namespace torch_mlu
