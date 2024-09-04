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

// CAUTION:
// The interfaces within this header file are deprecated and will be removed in
// future versions. Please replace with the corresponding interfaces in
// cnnlTensorDesc.h.

#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <vector>
#include "aten/utils/exceptions.h"
#include "aten/utils/tensor_util.h"
#include "cnnl.h" //  NOLINT

namespace torch_mlu {

class TORCH_MLU_API CnnlTensorDescriptor {
 public:
  // Init create Tensor descriptor
  CnnlTensorDescriptor();

  // Move constructor
  CnnlTensorDescriptor(CnnlTensorDescriptor&& other);

  // Disallow copy and assign operation
  CnnlTensorDescriptor(const CnnlTensorDescriptor& obj) = delete;
  void operator=(const CnnlTensorDescriptor&) = delete;

  ~CnnlTensorDescriptor();

  // set descriptor from tensor
  void set(const at::Tensor& t);

  void set(
      const at::Tensor& t,
      cnnlTensorLayout_t layout,
      cnnlDataType_t data_type = CNNL_DTYPE_INVALID);

  void set(const at::Tensor& t, cnnlDataType_t dtype);

  // an fast setter for array layout
  void set_array(const at::Tensor& t, cnnlDataType_t data_type);

  void set_onchip_dtype(cnnlDataType_t data_type);

  // for dimension conbined, just support 0<=dim <=3;
  // dim == 0 or 1, will combine dim 1 and dim 2 to new shape dim 1;
  // dim == 2, will combine dim 2 and dim 3 to new shape dim 2;
  // dim == 3, will combine dim 0 and dim 1 to new shape dim 0;
  // Just support channel_last format type.
  void set(
      const at::Tensor& t,
      const std::vector<int64_t>& tensor_cnnl_size,
      int64_t dim);

  void set_dim(const at::Tensor& t);

  void set_reduce(const at::Tensor& t);

  // TODO(CNNLCORE-13916) : delete after cnnl support.
  void set_reduce(
      const cnnlDataType_t& cnnl_dtype,
      const std::vector<int64_t>& keepdim);

  template <
      typename T,
      typename = std::enable_if_t<
          std::is_same<typename std::decay<T>::type, int64_t>::value ||
              std::is_same<typename std::decay<T>::type, int>::value,
          int>>
  void set(
      const at::Tensor& t,
      const std::vector<T>& shape_info,
      const std::vector<T>& stride_info,
      cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY,
      cnnlDataType_t data_type = CNNL_DTYPE_INVALID) {
    const int64_t t_dim = shape_info.size();
    TORCH_CHECK(
        t_dim == stride_info.size(), "shape size need equal to stride size.");

    // data_type default value is CNNL_DTYPE_INVALID in this interface,
    // and can't transmit to cnnl. so call cnnl interface will using
    // tensor dtype value when data_type value is default.
    auto* tensor_impl = getMluTensorImpl(t);
    if (data_type == CNNL_DTYPE_INVALID) {
      data_type = getCnnlType(tensor_impl);
    }
    auto setTensorDesc = [&](const int64_t dim,
                             const int64_t* size,
                             const int64_t* stride) -> void {
      TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(
          mut_desc(), layout, data_type, dim, size, stride));
    };
    if (!t_dim) {
      int64_t dim_array[1] = {1};
      setTensorDesc(1, dim_array, dim_array);
      return;
    }
    if (std::is_same<typename std::decay<T>::type, int64_t>::value == true) {
      setTensorDesc(t_dim, shape_info.data(), stride_info.data());
    } else {
      std::vector<int64_t> real_shape_info;
      std::vector<int64_t> real_stride_info;
      real_shape_info.reserve(t_dim);
      real_stride_info.reserve(t_dim);
      for (int i = 0; i < t_dim; i++) {
        real_shape_info.push_back(shape_info[i]);
        real_stride_info.push_back(stride_info[i]);
      }

      setTensorDesc(t_dim, real_shape_info.data(), real_stride_info.data());
    }
  }

  // This is the actual descriptor initializer
  // Maybe fix this in the future.
  cnnlTensorDescriptor_t mut_desc();

  cnnlTensorDescriptor_t desc() const {
    return desc_;
  }

 private:
  cnnlTensorDescriptor_t desc_ = nullptr;
};

} // end of namespace torch_mlu
