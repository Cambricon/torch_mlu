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

#include "aten/operators/cnnl/internal/foreach_common_utils.h"

namespace torch_mlu::ops {

/**
 * Note [ForeachLerpTensorsOp]
 * cnnl_foreach_lerp_tensors_op support three modes according to pytorch
 * gpu.
 * 1) tensor_list mode: aten/src/ATen/native/cuda/ForeachTernaryOp.cu
 *    all three tensor list are used, scalar_list is
 *    empty and scalar is not used;
 * 2) scalar mode: aten/src/ATen/native/cuda/ForeachTernaryOp.cu
 *    only left two tensor lists are used, right tensor list and scalar list are
 *    empty, scalar is used for each left tensor, and scalar type promote to
 *    op math type in torch-mlu side;
 * 3) scalar list mode: aten/src/ATen/native/cuda/ForeachTernaryOp.cu
 *    only left two tensor lists are used, each item in scalar list is used for
 * each tensor in left tensor list, and each value in scalar list will promote
 * to op match type in torch-mlu side. right tensor list is empty, and scalar is
 *    not used;
 */

template <typename scalar_t, bool isInplace>
void cnnl_foreach_lerp_op(
    at::TensorList tensors1,
    at::TensorList tensors2,
    at::TensorList tensors3, // mode 1
    at::TensorList outputs,
    const at::ArrayRef<at::Scalar>& scalar_list, // mode 2, not support yet
    const scalar_t& scalar, // mode 3
    const cnnlForeachLerpMode_t& mode) {
  auto stream = torch_mlu::getCurrentMLUStream();
  auto handle = getCurrentHandle();
  ForeachOPTensorScalarHandle<3, 1, isInplace, /*isReduceOp*/ false, scalar_t>
      tensor_desc_ptr({tensors1, tensors2, tensors3, outputs}, scalar_list);
  const int64_t tensor_num = tensor_desc_ptr.get_tensor_num();
  if (tensor_num == 0)
    return;
  auto [input1_desc_array, input1_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<0>();
  auto [input2_desc_array, input2_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<1>();
  // Only mode 1 will get real pointer, others are nullptr.
  auto [input3_desc_array, input3_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<2>();
  auto [output_desc_array, output_ptr_array] =
      tensor_desc_ptr.template get_output_tensor_desc_and_ptr<0>();

  tensorDescPtr_t scalar_desc = nullptr;
  // Only mode 2 will get real pointer, others are nullptr.
  scalar_t* scalar_ptr = tensor_desc_ptr.get_scalar_list_ptr();
  // get a cpu tensor desc
  if (mode == cnnlForeachLerpMode_t::FOREACH_LERP_SCALAR ||
      mode == cnnlForeachLerpMode_t::FOREACH_LERP_TENSOR_LIST) {
    c10::ScalarType type = c10::CppTypeToScalarType<scalar_t>::value;
    scalar_desc = std::move(
        getCpuTensorDesc(getCnnlDataType(type), CNNL_POINTER_MODE_HOST));
  }
  // mode 3
  if (mode == cnnlForeachLerpMode_t::FOREACH_LERP_SCALAR) {
    scalar_ptr = const_cast<scalar_t*>(&scalar);
  }

  size_t workspace_size = 0;
  at::DataPtr workspace_ptr;
  at::DataPtr cpu_workspace_ptr;
  TORCH_CNNL_CHECK(cnnlGetForeachLerpExtraInputSize(
      handle, mode, tensor_num, input1_desc_array, &workspace_size));
  if (workspace_size != 0) {
    // TODO(PYTORCH-12852): Foreach op not support graph now.
    TORCH_CHECK(
        torch_mlu::currentStreamCaptureStatusMayInitCtx() ==
            torch_mlu::CaptureStatus::None,
        "Foreach op not support graph now.");
    workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

    cpu_workspace_ptr = torch_mlu::HostAlloc(workspace_size);
    TORCH_CNNL_CHECK(cnnlInitForeachLerpExtraInput(
        handle,
        mode,
        tensor_num,
        input1_desc_array,
        input1_ptr_array,
        input2_ptr_array,
        input3_ptr_array,
        output_desc_array,
        output_ptr_array,
        cpu_workspace_ptr.get()));

    TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
        workspace_ptr.get(),
        cpu_workspace_ptr.get(),
        workspace_size,
        stream.stream(),
        CNRT_MEM_TRANS_DIR_HOST2DEV));
    MLUCachingHostAllocator_recordEvent(
        cpu_workspace_ptr.get(), cpu_workspace_ptr.get_context(), stream);
  }

  TORCH_CNNL_CHECK(cnnlForeachLerp(
      handle,
      mode,
      tensor_num,
      input1_desc_array,
      input1_ptr_array,
      input2_desc_array,
      input2_ptr_array,
      input3_desc_array,
      mode == cnnlForeachLerpMode_t::FOREACH_LERP_TENSOR_LIST
          ? input3_ptr_array
          : reinterpret_cast<const void* const*>(&scalar_ptr),
      workspace_ptr.get(),
      workspace_size,
      output_desc_array,
      output_ptr_array));
}

#define FOREACH_LERP_OP(scalar_t, _)                   \
  template void cnnl_foreach_lerp_op<scalar_t, true>(  \
      at::TensorList tensors1,                         \
      at::TensorList tensors2,                         \
      at::TensorList tensors3,                         \
      at::TensorList outputs,                          \
      const at::ArrayRef<at::Scalar>& scalar_list,     \
      const scalar_t& scalar,                          \
      const cnnlForeachLerpMode_t& mode);              \
  template void cnnl_foreach_lerp_op<scalar_t, false>( \
      at::TensorList tensors1,                         \
      at::TensorList tensors2,                         \
      at::TensorList tensors3,                         \
      at::TensorList outputs,                          \
      const at::ArrayRef<at::Scalar>& scalar_list,     \
      const scalar_t& scalar,                          \
      const cnnlForeachLerpMode_t& mode);

// Add almost pytorch scalar_t in here, and each op using self scalar type list
// in eachside.
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(FOREACH_LERP_OP)

#undef FOREACH_LERP_OP

} // namespace torch_mlu::ops
