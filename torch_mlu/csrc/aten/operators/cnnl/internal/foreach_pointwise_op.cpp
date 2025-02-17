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

template <typename scalar_t, bool isInplace>
void cnnl_foreach_pointwise_op(
    at::TensorList input,
    at::TensorList tensors1,
    at::TensorList tensors2,
    at::TensorList res_vec,
    const scalar_t& scalar, // scalar mode
    const at::ArrayRef<at::Scalar>& scalars, // scalar list mode
    const cnnlForeachOpMode_t& op_mode,
    const cnnlForeachPointWiseMode_t& scalar_mode) {
  auto stream = torch_mlu::getCurrentMLUStream();
  auto handle = getCurrentHandle();
  ForeachOPTensorScalarHandle<3, 1, isInplace, /*isReduceOp*/ false, scalar_t>
      tensor_desc_ptr({input, tensors1, tensors2, res_vec}, scalars);
  const int64_t tensor_num = tensor_desc_ptr.get_tensor_num();
  if (tensor_num == 0)
    return;
  auto [input_desc_array, input_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<0>();
  auto [tensors1_desc_array, tensors1_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<1>();
  auto [tensors2_desc_array, tensors2_ptr_array] =
      tensor_desc_ptr.template get_input_tensor_desc_and_ptr<2>();
  auto [output_desc_array, output_ptr_array] =
      tensor_desc_ptr.template get_output_tensor_desc_and_ptr<0>();

  tensorDescPtr_t scalar_desc = nullptr;
  // real ptr will be get in scalar list mode, otherwise a nullptr will be get
  scalar_t* scalar_ptr = tensor_desc_ptr.get_scalar_list_ptr();
  if (scalar_mode == cnnlForeachPointWiseMode_t::FOREACH_POINTWISE_SCALAR) {
    scalar_ptr = const_cast<scalar_t*>(&scalar);
  }

  TORCH_CNNL_CHECK(cnnlForeachPointwiseOp(
      handle,
      op_mode,
      scalar_mode,
      tensor_num,
      input_desc_array,
      input_ptr_array,
      tensors1_desc_array,
      tensors1_ptr_array,
      tensors2_desc_array,
      tensors2_ptr_array,
      (void*)scalar_ptr,
      output_desc_array,
      output_ptr_array));
}

#define FOREACH_POINTWISE_OP(scalar_t, _)                                \
  template void cnnl_foreach_pointwise_op<scalar_t, true>(               \
      at::TensorList input,                                              \
      at::TensorList tensors1,                                           \
      at::TensorList tensors2,                                           \
      at::TensorList res_vec,                                            \
      const scalar_t& scalar, /* for scalar mode*/                       \
      const at::ArrayRef<at::Scalar>& scalars, /* forscalar list mode */ \
      const cnnlForeachOpMode_t& op_mode,                                \
      const cnnlForeachPointWiseMode_t& scalar_mode);                    \
  template void cnnl_foreach_pointwise_op<scalar_t, false>(              \
      at::TensorList input,                                              \
      at::TensorList tensors1,                                           \
      at::TensorList tensors2,                                           \
      at::TensorList res_vec,                                            \
      const scalar_t& scalar, /* for scalar mode*/                       \
      const at::ArrayRef<at::Scalar>& scalars, /* forscalar list mode */ \
      const cnnlForeachOpMode_t& op_mode,                                \
      const cnnlForeachPointWiseMode_t& scalar_mode);

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(
    FOREACH_POINTWISE_OP)

#undef FOREACH_POINTWISE_OP

} // namespace torch_mlu::ops
