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
#include "aten/utils/tensor_util.h"

namespace torch_mlu::ops {

/**
 * Note [ForeachBinaryTensorsOp]
 * cnnl_foreach_binary_tensors_op support four modes according to pytorch
 * gpu.
 * 1) tensor_list mode: aten/src/ATen/native/cuda/ForeachBinaryOpList.cu
 *    left and right tensor list is equal and with alpha, scalar_list is
 *    empty and scalar is not used;
 * 2) scalar list mode: aten/src/ATen/native/cuda/ForeachBinaryOpScalarList.cu
 *    only left tensor list is used, each item in scalar list is used for each
 *    tensor in left tensor list, and each value in scalar list will promote to
 *    op match type in torch-mlu side. right tensor list is empty, and scalar is
 *    not used;
 * 3) scalar mode: aten/src/ATen/native/cuda/ForeachBinaryOpScalar.cu
 *    only left tensor list is used, right tensor list and scalar list are
 *    empty, scalar is used for each left tensor, and scalar type promote to
 *    op math type in torch-mlu side;
 * 4) scalar tensor mode:
 * aten/src/ATen/native/cuda/ForeachBinaryOpScalarTensor.cu Now only used for
 * mul op. Similar to scalar mode, but the scalar is device scalar tensor, and
 * scalar type is same with left tensors type, need cnnl kernel to promote type
 * to op math type. (Not using now)
 *
 */

template <typename scalar_t, bool isInplace>
void cnnl_foreach_binary_tensors_op(
    at::TensorList tensors1,
    at::TensorList tensors2, // mode 1
    at::TensorList outputs,
    const at::ArrayRef<at::Scalar>& scalar_list, // mode 2
    const scalar_t& scalar, // mode 3
    const at::Tensor& scalar_tensor, // mode 4
    const scalar_t& alpha,
    const cnnlForeachOpMode_t& op_mode,
    const cnnlForeachBinaryMode_t& mode) {
  auto handle = getCurrentHandle();
  ForeachOPTensorScalarHandle<2, 1, isInplace, /*isReduceOp*/ false, scalar_t>
      tensor_desc_ptr({tensors1, tensors2, outputs}, scalar_list);
  const int64_t tensor_num = tensor_desc_ptr.get_tensor_num();
  if (tensor_num == 0)
    return;
  auto [input1_desc_array, input1_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<0>();
  // Only mode 1 will get real pointer, others are nullptr.
  auto [input2_desc_array, input2_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<1>();
  auto [output_desc_array, output_ptr_array] =
      tensor_desc_ptr.template get_output_tensor_desc_and_ptr<0>();

  tensorDescPtr_t scalar_desc = nullptr;
  // Only mode 2 will get real pointer, others are nullptr.
  scalar_t* scalar_ptr = tensor_desc_ptr.get_scalar_list_ptr();
  // get a cpu tensor desc
  if (mode == cnnlForeachBinaryMode_t::FOREACH_BINARY_SCALAR ||
      mode == cnnlForeachBinaryMode_t::FOREACH_BINARY_SCALAR_LIST) {
    c10::ScalarType type = c10::CppTypeToScalarType<scalar_t>::value;
    scalar_desc = std::move(
        getCpuTensorDesc(getCnnlDataType(type), CNNL_POINTER_MODE_HOST));
  }
  // mode 3
  if (mode == cnnlForeachBinaryMode_t::FOREACH_BINARY_SCALAR) {
    scalar_ptr = const_cast<scalar_t*>(&scalar);
  } else if (mode == cnnlForeachBinaryMode_t::FOREACH_BINARY_SCALAR_TENSOR) {
    auto scalar_tensor_impl = getMluTensorImpl(scalar_tensor);
    scalar_desc = getTensorDesc(scalar_tensor_impl);
    scalar_ptr = static_cast<scalar_t*>(mlu_data_ptr(scalar_tensor_impl));
  }

  size_t workspace_size = 0;
  at::DataPtr workspace_ptr;
  TORCH_CNNL_CHECK(cnnlGetForeachBinaryOpWorkspaceSize(
      handle, tensor_num, input1_desc_array, &workspace_size));
  if (workspace_size != 0) {
    workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  }

  TORCH_CNNL_CHECK(cnnlForeachBinaryOp(
      handle,
      op_mode,
      mode,
      tensor_num,
      input1_desc_array,
      input1_ptr_array,
      input2_desc_array,
      input2_ptr_array,
      scalar_desc.get(),
      (void*)scalar_ptr,
      mode == cnnlForeachBinaryMode_t::FOREACH_BINARY_TENSOR_LIST ? &alpha
                                                                  : nullptr,
      workspace_ptr.get(),
      workspace_size,
      output_desc_array,
      output_ptr_array));
}

#define FOREACH_BINARY_OP(scalar_t, _)                           \
  template void cnnl_foreach_binary_tensors_op<scalar_t, true>(  \
      at::TensorList tensors1,                                   \
      at::TensorList tensors2,                                   \
      at::TensorList outputs,                                    \
      const at::ArrayRef<at::Scalar>& scalar_list,               \
      const scalar_t& scalar,                                    \
      const at::Tensor& scalar_tensor,                           \
      const scalar_t& alpha,                                     \
      const cnnlForeachOpMode_t& op_mode,                        \
      const cnnlForeachBinaryMode_t& mode);                      \
  template void cnnl_foreach_binary_tensors_op<scalar_t, false>( \
      at::TensorList tensors1,                                   \
      at::TensorList tensors2,                                   \
      at::TensorList outputs,                                    \
      const at::ArrayRef<at::Scalar>& scalar_list,               \
      const scalar_t& scalar,                                    \
      const at::Tensor& scalar_tensor,                           \
      const scalar_t& alpha,                                     \
      const cnnlForeachOpMode_t& op_mode,                        \
      const cnnlForeachBinaryMode_t& mode);

// Add almost pytorch scalar_t in here, and each op using self scalar type list
// in eachside.
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(FOREACH_BINARY_OP)

#undef FOREACH_BINARY_OP

} // namespace torch_mlu::ops
