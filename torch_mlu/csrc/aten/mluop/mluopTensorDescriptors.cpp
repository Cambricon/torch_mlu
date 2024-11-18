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
#include "aten/mluop/mluopUtils.h"
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

void MluOpTensorDescriptor::set(
    const at::Tensor& t,
    mluOpTensorLayout_t layout,
    mluOpDataType_t data_type) {
  // set cpu scalar
  if (isCpuScalar(t)) {
    if (data_type == MLUOP_DTYPE_INVALID) {
      data_type = getMluOpDataType(t.dtype());
    }
    TORCH_MLUOP_CHECK(mluOpSetTensorDescriptorPointerMode(
        this->mut_desc(), MLUOP_POINTER_MODE_HOST));
    TORCH_MLUOP_CHECK(mluOpSetTensorDescriptorEx_v2(
        this->mut_desc(),
        layout,
        data_type,
        0 /*dimNb*/,
        nullptr /*dimSize*/,
        nullptr /*dimStride*/));
    return;
  }
  int t_dim = t.dim();
  // TODO(PYTORCH-11599) : Utimately should remove all getCnnlType related code
  // TODO(tonghengwen) : This getMluTensorImpl need some rework.
  auto* tensor_impl = getMluTensorImpl(t);
  if (data_type == MLUOP_DTYPE_INVALID) {
    data_type = cnnlTypeToMluOpType(getCnnlType(tensor_impl));
  }

  if (!t_dim) {
    t_dim = 1;
    std::vector<int64_t> dim_array(1, 1);
    TORCH_MLUOP_CHECK(mluOpSetTensorDescriptorEx_v2(
        this->mut_desc(),
        MLUOP_LAYOUT_ARRAY,
        data_type,
        t_dim,
        dim_array.data(),
        dim_array.data()));
    return;
  }
  std::vector<int64_t> shape_info(t.sizes().vec());
  std::vector<int64_t> stride_info(t.strides().vec());
  std::vector<int64_t> cnnl_stride_info(t_dim);
  if (t.is_contiguous()) {
    cnnl_stride_info = get_contiguous_strides(t.sizes());
  } else {
    cnnl_stride_info = get_cnnl_strides(shape_info, stride_info);
  }

  if (layout == MLUOP_LAYOUT_NHWC || layout == MLUOP_LAYOUT_NDHWC ||
      layout == MLUOP_LAYOUT_NLC) {
    convertShapeAndStride(shape_info, cnnl_stride_info);
  } else if (layout == MLUOP_LAYOUT_HWCN) {
    auto convertDepthWiseConvShapeStride =
        [](const std::vector<int64_t>& vec,
           std::vector<int64_t>& target_vec,
           std::vector<int64_t>& stride_vec) {
          // NCHW --> HWCN
          target_vec[0] = static_cast<int>(vec[2]);
          target_vec[1] = static_cast<int>(vec[3]);
          target_vec[2] = static_cast<int>(vec[1]);
          target_vec[3] = static_cast<int>(vec[0]);
          // Caculate Stride just like contiguous of HWCN.
          stride_vec[3] = 1;
          stride_vec[2] = target_vec[3] * stride_vec[3];
          stride_vec[1] = target_vec[2] * stride_vec[2];
          stride_vec[0] = target_vec[1] * stride_vec[1];
        };
    convertDepthWiseConvShapeStride(
        t.sizes().vec(), shape_info, cnnl_stride_info);
  }
  TORCH_MLUOP_CHECK(mluOpSetTensorDescriptorEx_v2(
      this->mut_desc(),
      layout,
      data_type,
      t_dim,
      shape_info.data(),
      cnnl_stride_info.data()));
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
