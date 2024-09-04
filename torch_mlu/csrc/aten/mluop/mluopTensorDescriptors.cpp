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

#include "aten/mluop/mluopTensorDescriptors.h"
#include "framework/core/tensor_impl.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/utils.h"
#include "aten/utils/exceptions.h"

namespace torch_mlu {

void MluOpTensorDescriptor::set(const at::Tensor& t) {
  mluOpDataType_t data_type = getMluOpDataType(t.dtype());
  mluOpTensorLayout_t layout = MLUOP_LAYOUT_ARRAY; // ARRAY AS DEFAULT
  int t_dim = t.dim();
  std::vector<int64_t> dim_array;
  if (t_dim == 0) {
    dim_array.push_back(1);
  } else {
    auto t_vec = t.sizes().vec();
    for (int i = 0; i < t_dim; i++) {
      dim_array.push_back(t_vec[i]);
    }
  }
  set_impl(t, layout, data_type, dim_array);
}

// protected:
void MluOpTensorDescriptor::set_impl(
    const at::Tensor& t,
    mluOpTensorLayout_t layout,
    mluOpDataType_t dtype,
    std::vector<int64_t>& dims) {
  int dimNb = dims.size();
  auto dim_int32 = checkUpperBoundAndCastTo<int>(dims);
  TORCH_MLUOP_CHECK(mluOpSetTensorDescriptor(
      this->mut_desc(), layout, dtype, dimNb, dim_int32.data()));
}

} // namespace torch_mlu
