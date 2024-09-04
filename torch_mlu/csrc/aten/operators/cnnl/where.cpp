/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/utils/cnnl_util.h"
#include "aten/utils/dispatch.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/binaryops_util.h"
#include "aten/utils/internal_util.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {
namespace ops {

void where_kernel_impl(at::TensorIterator& iter) {
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "where");
  auto output = iter.output(0);
  AT_DISPATCH_ALL_MLU_TYPES_AND_HALF_AND_BFLOAT16_EXCEPT_UINT8_AND_BOOL(
      iter.dtype(), "cnnl_where", [&] {
        cnnl_where_internal(
            output, iter.input(0), iter.input(1), iter.input(2));
      });
  iter.cast_outputs();
}

template <typename... Args>
Device out_device(Args&... inps) {
  for (const auto& i : {inps...}) {
    if (i.device() != at::kCPU) {
      return i.device();
    }
  }
  return at::kCPU;
}

at::Tensor& cnnl_where_out(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  Tensor self_, other_, condition_;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_ = self.to(result_type);
    other_ = other.to(result_type);
  } else {
    self_ = self;
    other_ = other;
  }
  auto device = out_device(condition, self_, other_);
  condition_ = condition;
  if (device != at::kCPU) { // allow CPU scalars on non-cpu device
    if (condition.device() != device && condition.ndimension() == 0) {
      condition_ = condition.to(device);
    }
    if (self_.device() != device && self_.ndimension() == 0) {
      self_ = self_.to(device);
    }
    if (other_.device() != device && other_.ndimension() == 0) {
      other_ = other_.to(device);
    }
  }
  if (condition.scalar_type() == c10::ScalarType::Byte) {
    TORCH_WARN_ONCE(
        "where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
  } else {
    TORCH_CHECK(
        condition.scalar_type() == c10::ScalarType::Bool,
        "where expected condition to be a boolean tensor, but got a tensor with dtype ",
        condition.scalar_type());
  }
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(out)
                  .add_input(condition_)
                  .add_input(self_)
                  .add_input(other_)
                  .build();
  where_kernel_impl(iter);
  return out;
}

at::Tensor cnnl_where(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  auto device = out_device(condition, self, other);
  auto result_type = at::native::result_type(self, other);
  Tensor ret = at::empty({0}, self.options().dtype(result_type).device(device));
  cnnl_where_out(condition, self, other, ret);
  return ret;
}

} // namespace ops
} // namespace torch_mlu
