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

} // namespace torch_mlu::ops
