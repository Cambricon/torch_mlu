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

#include "cnnl.h" //  NOLINT
#include "c10/core/TensorImpl.h"

namespace torch_mlu::ops {

#define CNNL_MAX_DIM_SIZE 8

inline cnnlTensorLayout_t suggestCnnlLayout(const c10::TensorImpl* self) {
  cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
  switch (self->dim()) {
    case 4:
      layout = (self->is_contiguous(at::MemoryFormat::ChannelsLast))
          ? CNNL_LAYOUT_NHWC
          : CNNL_LAYOUT_NCHW;
      break;
    case 5:
      layout = (self->is_contiguous(at::MemoryFormat::ChannelsLast3d))
          ? CNNL_LAYOUT_NDHWC
          : CNNL_LAYOUT_NCDHW;
      break;
  }
  return layout;
}

inline cnnlTensorLayout_t suggestCnnlLayout(
    const at::MemoryFormat& memory_format) {
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    return CNNL_LAYOUT_NHWC;
  } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
    return CNNL_LAYOUT_NDHWC;
  } else {
    return CNNL_LAYOUT_ARRAY;
  }
}

struct CnnlDescriptorDeleter {
  void operator()(cnnlTensorStruct* ptr) {
    if (ptr != nullptr) {
      TORCH_CNNL_CHECK(cnnlDestroyTensorDescriptor(ptr));
    }
  }
};

using tensorDescPtr_t =
    std::unique_ptr<cnnlTensorStruct, CnnlDescriptorDeleter>;

/**
 * Note [CATCH Tensor Desc Suggest]
 * 1) Common suggest:
 * 1.1) If all tensors type are same in op internal function, call getCnnlType
 *      first to get tensor type, then achieve each tensor desc by call
 * getTensorDesc API with tensor type; Using getTensorDesc(const
 * c10::TensorImpl* self, cnnlDataType_t data_type). 1.2) If all tensors layout
 * are same in op internal function, call suggestCnnlLayout first to get tensor
 * layout, then achieve each tensor desc by call getTensorDesc API with tensor
 * layout; Using getTensorDesc(const c10::TensorImpl* self, cnnlTensorLayout_t
 * layout). 1.3) If all tensors layout and type are same in op internal
 * function, Using API getTensorDesc(const c10::TensorImpl* self, cnnlDataType_t
 * data_type, cnnlTensorLayout_t layout). 1.4) Some op need set on-chip type for
 * promote type when kernel calculate, Using API getTensorDesc(const
 * c10::TensorImpl* self, cnnlDataType_t data_type, cnnlTensorLayout_t layout,
 * cnnlDataType_t on_chip_type).
 *
 * 2) Op support stride: Using getTensorDesc and to specify the tensor layout
 *    to CNNL_LAYOUT_ARRAY;
 *
 * 3) OP take dim info: Need keep dim and tensors memory_format or
 * cnnlTensorLayout_t same. This mean using same memory_format value or
 * cnnlTensorLayout_t value in function getTensorDesc API and
 * modify_dim_based_on_layout API.
 *
 * 4) Element-wise op: Using getTensorDescAndCoalesceDims directly.
 *
 * 5) If tensors are very different, using different getTensorDesc API for each
 * tensor.
 *
 */

// Get all desc info based on input tensor impl.
// Using size, stride and dtype info from input tensor impl;
// Get the corresponding tensor layout through function suggestCnnlLayout;
// on-chip dtype is CNNL_DTYPE_INVALID.
tensorDescPtr_t getTensorDesc(const c10::TensorImpl* self);

// Using specific tensor dtype and input tensor impl to create a desc.
// which size, stride info and tensor layout is based on input tensor impl;
// Get the corresponding tensor layout through function suggestCnnlLayout;
// on-chip dtype is CNNL_DTYPE_INVALID.
tensorDescPtr_t getTensorDesc(
    const c10::TensorImpl* self,
    cnnlDataType_t data_type);

// Using specific tensor layout and input tensor impl to create a desc.
// Size, stride info and dtype is based on input tensor impl;
// on-chip dtype is CNNL_DTYPE_INVALID.
tensorDescPtr_t getTensorDesc(
    const c10::TensorImpl* self,
    cnnlTensorLayout_t layout);

// Using specific off-chip dtype, tensor layout and
// input tensor impl to create a desc.
// Size, stride info is based on input tensor impl;
// on-chip dtype is CNNL_DTYPE_INVALID.
tensorDescPtr_t getTensorDesc(
    const c10::TensorImpl* self,
    cnnlDataType_t data_type,
    cnnlTensorLayout_t layout);

// Using specific off-chip dtype, tensor layout and on-chip
// dtype to create a desc, and size, stride info is based on
// input tensor impl.
tensorDescPtr_t getTensorDesc(
    const c10::TensorImpl* self,
    cnnlDataType_t data_type,
    cnnlTensorLayout_t layout,
    cnnlDataType_t on_chip_type);

// Using outside size and stride to create a new desc.
tensorDescPtr_t getTensorDesc(
    const at::IntArrayRef& shape_info,
    const at::IntArrayRef& stride_info,
    cnnlDataType_t data_type,
    cnnlTensorLayout_t layout,
    cnnlDataType_t on_chip_type = CNNL_DTYPE_INVALID);

// Now only used for softmax.
tensorDescPtr_t getTensorDescWithKeepDimAndOnlyThreeDims(
    const c10::TensorImpl* self,
    int64_t keep_dim,
    cnnlTensorLayout_t layout);

// Used for element-wise op, all size and stride coalesced to one dim, which
// size value equal to self->numel(), stride value equal to 1.
tensorDescPtr_t getTensorDescAndCoalesceDims(const c10::TensorImpl* self);

// Get cpu tensor desc.
tensorDescPtr_t getCpuTensorDesc(
    cnnlDataType_t data_type,
    cnnlPointerMode_t pointer_mode = CNNL_POINTER_MODE_DEVICE);

} // namespace torch_mlu::ops
