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

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <vector>
#include "aten/utils/exceptions.h"
#include "aten/utils/tensor_util.h"
#include "cnnl.h" //  NOLINT

namespace torch_mlu {

template <typename T, cnnlStatus_t (*dtor)(T*)>
struct CnnlDescriptorDeleter {
  void operator()(T* ptr) {
    if (ptr != nullptr) {
      TORCH_CNNL_CHECK(dtor(ptr));
    }
  }
};

template <typename T, cnnlStatus_t (*ctor)(T**), cnnlStatus_t (*dtor)(T*)>
class CnnlDescriptor {
 public:
  CnnlDescriptor() = default;

  // Use desc() to access the underlying descriptor pointer in
  // a read-only fashion.  Most client code should use this.
  // If the descriptor was never initialized, this will return
  // nullptr.
  T* desc() const {
    return desc_.get();
  }
  T* desc() {
    return desc_.get();
  }

  // Use CnnlDescriptor() to access the underlying desciptor pointer
  // if you intend to modify what it points to This will ensure
  // that the descriptor is initialized.
  // Code in this file will use this function.
  T* mut_desc() {
    init();
    return desc_.get();
  }

 protected:
  void init() {
    if (desc_ == nullptr) {
      T* ptr;
      TORCH_CNNL_CHECK(ctor(&ptr));
      desc_.reset(ptr);
    }
  }

 private:
  std::unique_ptr<T, CnnlDescriptorDeleter<T, dtor>> desc_;
};

std::vector<int64_t> get_cnnl_strides(
    std::vector<int64_t> shape_info,
    std::vector<int64_t> stride_info);

// modify tensor size and stride order based on
// channels_first to channels_last or channels_last_3d.
// which this is not same with pytorch original layout,
// this real layout is based on data storage real order.
// example: modify channels_last tensor dim to cnnl nhwc tensor desc.
//            N    C H W  -->   N    H W C
//          C*H*W  1 W C  --> C*H*W  W C 1
template <typename T>
void convertShapeAndStride(
    std::vector<T>& shape_info,
    std::vector<T>& stride_info) {
  TORCH_MLU_CHECK(
      shape_info.size() == stride_info.size(),
      "shape size need equal to stride size.");
  const int dim = shape_info.size();
  std::vector<T> temp_shape_info(dim);
  std::vector<T> temp_stride_info(dim);
  temp_shape_info[0] = shape_info[0];
  temp_stride_info[0] = stride_info[0];
  for (size_t i = 0; i < dim - 1; ++i) {
    const int index = (i + 1) % (dim - 1) + 1;
    temp_shape_info[i + 1] = shape_info[index];
    temp_stride_info[i + 1] = stride_info[index];
  }
  shape_info.assign(temp_shape_info.begin(), temp_shape_info.end());
  stride_info.assign(temp_stride_info.begin(), temp_stride_info.end());
}

} // end of namespace torch_mlu
