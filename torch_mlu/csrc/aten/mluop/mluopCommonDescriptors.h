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

#include <ATen/ATen.h>
#include <vector>
#include "mlu_op.h"
#include "aten/utils/exceptions.h"

namespace torch_mlu {

template <typename T, mluOpStatus_t (*dtor)(T*)>
struct MluOpDescriptorDeleter {
  void operator()(T* ptr) {
    if (ptr != nullptr) {
      TORCH_MLUOP_CHECK(dtor(ptr));
    }
  }
};

template <typename T, mluOpStatus_t (*ctor)(T**), mluOpStatus_t (*dtor)(T*)>
class MluOpDescriptor {
 public:
  MluOpDescriptor() = default;

  T* desc() const {
    return desc_.get();
  }
  T* desc() {
    return desc_.get();
  }

  T* mut_desc() {
    init();
    return desc_.get();
  }

 protected:
  void init() {
    if (desc_ == nullptr) {
      T* ptr;
      TORCH_MLUOP_CHECK(ctor(&ptr));
      desc_.reset(ptr);
    }
  }

 private:
  std::unique_ptr<T, MluOpDescriptorDeleter<T, dtor>> desc_;
};

} // namespace torch_mlu

#endif
