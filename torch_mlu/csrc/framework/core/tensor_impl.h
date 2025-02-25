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

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <ATen/Functions.h>
#include "cnrt.h" // NOLINT
#include "framework/core/caching_allocator.h"

#include "aten/utils/exceptions.h"
#include "aten/utils/types.h"
#include "utils/common.h"

namespace torch_mlu {

class MLUTensorImpl : public c10::External {
 public:
  cnnlDataType_t cnnl_data_type_ = CNNL_DTYPE_INVALID;
  MLUTensorImpl() {}
  explicit MLUTensorImpl(const at::Tensor& t) {
    cnnl_data_type_ = getCnnlDataType(t.dtype());
  }
  ~MLUTensorImpl() {}
  std::unique_ptr<c10::External> clone() const override {
    auto impl = std::make_unique<MLUTensorImpl>();
    impl->cnnl_data_type_ = this->cnnl_data_type_;
    return impl;
  }
  void* data_ptr(void* data, int64_t offset) const override {
    auto on_chip_element_size = getCnnlTypeSize(cnnl_data_type_);
    return static_cast<void*>(
        static_cast<char*>(data) + on_chip_element_size * offset);
  }
};

} // namespace torch_mlu
