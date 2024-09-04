/*
All modification made by Cambricon Corporation: © 2024 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2024, the respective contributors
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

#include "aten/utils/exceptions.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/utils.h"
#include "aten/cnnl/cnnlTensorDesc.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu::ops {

static tensorDescPtr_t getZeroDimTensorDesc(cnnlDataType_t data_type) {
  constexpr std::array<int64_t, 1> sizes_and_strides_{1};
  cnnlTensorDescriptor_t desc;
  TORCH_CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(
      desc,
      CNNL_LAYOUT_ARRAY,
      data_type,
      1,
      sizes_and_strides_.data(),
      sizes_and_strides_.data()));
  tensorDescPtr_t ptr(desc);
  return ptr;
}

static tensorDescPtr_t getTensorDescWithAllValues(
    const at::IntArrayRef& shape_info,
    const at::IntArrayRef& stride_info,
    cnnlDataType_t data_type = CNNL_DTYPE_INVALID,
    cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY,
    cnnlDataType_t on_chip_type = CNNL_DTYPE_INVALID) {
  const int dim = shape_info.size();
  TORCH_MLU_CHECK(
      dim == stride_info.size(), "Length of shape and stride are not equal.");
  bool is_channel_last = ((layout == CNNL_LAYOUT_NHWC && dim == 4) ||
                          (layout == CNNL_LAYOUT_NDHWC && dim == 5) ||
                          (layout == CNNL_LAYOUT_NLC && dim == 3))
      ? true
      : false;
  cnnlTensorDescriptor_t desc;
  TORCH_CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  // Convert size and stride to contiguous.
  int64_t* size_dptr = const_cast<int64_t*>(shape_info.data());
  int64_t* stride_dptr = const_cast<int64_t*>(stride_info.data());
  std::vector<int64_t> shape_temp;
  std::vector<int64_t> stride_temp;
  if (is_channel_last == true) {
    shape_temp = std::move(shape_info.vec());
    stride_temp = std::move(stride_info.vec());
    // Swap is faster than memcpy or using vector operator[] circle.
    std::swap(shape_temp[1], shape_temp[2]);
    std::swap(stride_temp[1], stride_temp[2]);
    if (dim == 4) {
      std::swap(shape_temp[2], shape_temp[3]);
      std::swap(stride_temp[2], stride_temp[3]);
    } else if (dim == 5) {
      std::swap(shape_temp[2], shape_temp[3]);
      std::swap(shape_temp[3], shape_temp[4]);
      std::swap(stride_temp[2], stride_temp[3]);
      std::swap(stride_temp[3], stride_temp[4]);
    }
    size_dptr = shape_temp.data();
    stride_dptr = stride_temp.data();
  }
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(
      desc, layout, data_type, dim, size_dptr, stride_dptr));
  if (on_chip_type != CNNL_DTYPE_INVALID) {
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorOnchipDataType(desc, on_chip_type));
  }
  tensorDescPtr_t ptr(desc);
  return ptr;
}

tensorDescPtr_t getTensorDesc(const c10::TensorImpl* self) {
  if (self == nullptr)
    return nullptr;
  cnnlDataType_t data_type = getCnnlType(self);
  if (self->dim() == 0)
    return getZeroDimTensorDesc(data_type);
  cnnlTensorLayout_t layout = suggestCnnlLayout(self);
  return getTensorDescWithAllValues(
      self->sizes(), self->strides(), data_type, layout);
}

tensorDescPtr_t getTensorDesc(
    const c10::TensorImpl* self,
    cnnlTensorLayout_t layout) {
  if (self == nullptr)
    return nullptr;
  cnnlDataType_t data_type = getCnnlType(self);
  if (self->dim() == 0)
    return getZeroDimTensorDesc(data_type);
  return getTensorDescWithAllValues(
      self->sizes(), self->strides(), data_type, layout);
}

tensorDescPtr_t getTensorDesc(
    const c10::TensorImpl* self,
    cnnlDataType_t data_type) {
  if (self == nullptr)
    return nullptr;
  if (self->dim() == 0)
    return getZeroDimTensorDesc(data_type);
  cnnlTensorLayout_t layout = suggestCnnlLayout(self);
  return getTensorDescWithAllValues(
      self->sizes(), self->strides(), data_type, layout);
}

tensorDescPtr_t getTensorDesc(
    const c10::TensorImpl* self,
    cnnlDataType_t data_type,
    cnnlTensorLayout_t layout) {
  if (self == nullptr)
    return nullptr;
  if (self->dim() == 0)
    return getZeroDimTensorDesc(data_type);
  return getTensorDescWithAllValues(
      self->sizes(), self->strides(), data_type, layout);
}

tensorDescPtr_t getTensorDesc(
    const c10::TensorImpl* self,
    cnnlDataType_t data_type,
    cnnlTensorLayout_t layout,
    cnnlDataType_t on_chip_type) {
  if (self == nullptr)
    return nullptr;
  if (self->dim() == 0)
    return getZeroDimTensorDesc(data_type);
  return getTensorDescWithAllValues(
      self->sizes(), self->strides(), data_type, layout, on_chip_type);
}

tensorDescPtr_t getTensorDesc(
    const at::IntArrayRef& shape_info,
    const at::IntArrayRef& stride_info,
    cnnlDataType_t data_type,
    cnnlTensorLayout_t layout,
    cnnlDataType_t on_chip_type) {
  if (shape_info.size() == 0)
    return getZeroDimTensorDesc(data_type);
  return getTensorDescWithAllValues(
      shape_info, stride_info, data_type, layout, on_chip_type);
}

tensorDescPtr_t getTensorDescWithKeepDimAndOnlyThreeDims(
    const c10::TensorImpl* self,
    int64_t keep_dim,
    cnnlTensorLayout_t layout) {
  if (self == nullptr)
    return nullptr;
  const int input_dim = self->dim();
  cnnlDataType_t data_type = getCnnlType(self);
  if (input_dim == 0) {
    constexpr std::array<int64_t, 3> size_and_stride_{1, 1, 1};
    return getTensorDescWithAllValues(
        size_and_stride_, size_and_stride_, data_type, CNNL_LAYOUT_ARRAY);
  }
  std::vector<int64_t> shape_temp(std::move(self->sizes().vec()));
  std::vector<int64_t> stride_temp(std::move(self->strides().vec()));
  bool is_channel_last = ((layout == CNNL_LAYOUT_NHWC && input_dim == 4) ||
                          (layout == CNNL_LAYOUT_NDHWC && input_dim == 5))
      ? true
      : false;
  if (is_channel_last) {
    // Swap is faster than memcpy or using vector operator[] circle.
    std::swap(shape_temp[1], shape_temp[2]);
    std::swap(stride_temp[1], stride_temp[2]);
    std::swap(shape_temp[2], shape_temp[3]);
    std::swap(stride_temp[2], stride_temp[3]);
    if (input_dim == 5) {
      std::swap(shape_temp[3], shape_temp[4]);
      std::swap(stride_temp[3], stride_temp[4]);
    }
  }
  const int out_dim = 3;
  int64_t inner_size = 1;
  int64_t outer_size = 1;
  const int64_t dim_size = shape_temp[keep_dim];
  for (int64_t i = 0; i < keep_dim; ++i) {
    outer_size *= shape_temp[i];
  }
  for (int64_t i = keep_dim + 1; i < input_dim; ++i) {
    inner_size *= shape_temp[i];
  }
  // Get new size and stride.
  std::vector<int64_t> shape_info(out_dim, 1);
  std::vector<int64_t> stride_info(out_dim, 1);
  if (keep_dim == 0 && input_dim == 1) {
    shape_info[2] = dim_size;
  } else if (keep_dim == 0 && input_dim != 1) {
    // size [dim_size, inner_size, 1] and stride [inner_size, 1, 1]
    shape_info[0] = dim_size;
    shape_info[1] = inner_size;
    stride_info[0] = inner_size;
  } else if (keep_dim == input_dim - 1) {
    // size [1, outer_size, dim_size]
    // and stride [outer_size * dim_size, dim_size, 1]
    shape_info[1] = outer_size;
    shape_info[2] = dim_size;
    stride_info[0] = dim_size * outer_size;
    stride_info[1] = dim_size;
  } else {
    // size [outer_size, dim_size, inner_size]
    // and stride [dim_size * inner_size, inner_size, 1]
    shape_info[0] = outer_size;
    shape_info[1] = dim_size;
    shape_info[2] = inner_size;
    stride_info[0] = dim_size * inner_size;
    stride_info[1] = inner_size;
  }
  return getTensorDescWithAllValues(
      shape_info, stride_info, data_type, CNNL_LAYOUT_ARRAY);
}

// Used for element-wise op.
tensorDescPtr_t getTensorDescAndCoalesceDims(const c10::TensorImpl* self) {
  if (self == nullptr)
    return nullptr;
  TORCH_MLU_CHECK(
      self->is_non_overlapping_and_dense(),
      "Only support Tensor is non overlapping and dese.");
  cnnlDataType_t data_type = getCnnlType(self);
  if (self->dim() == 0)
    return getZeroDimTensorDesc(data_type);
  return getTensorDescWithAllValues(
      {self->numel()}, {1}, data_type, CNNL_LAYOUT_ARRAY);
}

tensorDescPtr_t getCpuTensorDesc(
    cnnlDataType_t data_type,
    cnnlPointerMode_t pointer_mode) {
  cnnlTensorDescriptor_t desc;
  TORCH_CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
  if (pointer_mode != CNNL_POINTER_MODE_DEVICE) {
    TORCH_CNNL_CHECK(cnnlSetTensorDescriptorPointerMode(desc, pointer_mode));
  }
  TORCH_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(
      desc, CNNL_LAYOUT_ARRAY, data_type, 0, nullptr, nullptr));
  tensorDescPtr_t ptr(desc);
  return ptr;
}

} // namespace torch_mlu::ops
