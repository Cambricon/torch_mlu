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

#pragma once
#ifdef USE_MLUOP

#include "aten/mluop/mluopCommonDescriptors.h"
#include "aten/mluop/mluopUtils.h"

namespace torch_mlu {

class TORCH_MLU_API MluOpTensorDescriptor : public MluOpDescriptor<
                                                mluOpTensorStruct,
                                                &mluOpCreateTensorDescriptor,
                                                &mluOpDestroyTensorDescriptor> {
 public:
  MluOpTensorDescriptor() = default;

  // Default Set:
  void set(const at::Tensor& t);
  // Specified Set:
  // use this api if the strides of tensor need to be set
  void set(
      const at::Tensor& t,
      mluOpTensorLayout_t layout,
      mluOpDataType_t data_type = MLUOP_DTYPE_INVALID);

  // set_dtype, set_layout etc.

 protected:
  /*
      Protected set is 1-1 to mluOpSetTensorDescriptors in mlu_op.h.
      TensorDescriptor 1->N Public.set(_sth) N->1 Protected.set_impl 1->1
     mluOpSet.
  */

  // mluOpSetTensorDescriptor(mluOpTensorDescriptor_t desc,
  //           layout  *-->      mluOpTensorLayout_t layout,
  //           dtype   *-->      mluOpDataType_t dtype,
  //           dims    +-->      int dimNb,
  //                   |-->      const int dimSize[]);
  void set_impl(
      const at::Tensor& t,
      mluOpTensorLayout_t layout,
      mluOpDataType_t dtype,
      std::vector<int64_t>& dims);
};

} // namespace torch_mlu

#endif
