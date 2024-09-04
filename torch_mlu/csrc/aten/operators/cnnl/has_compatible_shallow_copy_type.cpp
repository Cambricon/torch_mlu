/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
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
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

bool _has_compatible_shallow_copy_type(
    const at::Tensor& self,
    const at::Tensor& from) {
  c10::DispatchKeySet self_key = self.key_set();
  c10::DispatchKeySet from_key = from.key_set();
  auto is_dense = [](c10::DispatchKeySet ts) {
    constexpr auto dense_backends = c10::DispatchKeySet(
        {c10::BackendComponent::CPUBit, c10::BackendComponent::PrivateUse1Bit});
    constexpr auto dense_k = c10::DispatchKeySet(c10::DispatchKey::Dense);
    return ts.has_any(dense_k) && ts.has_any(dense_backends);
  };
  return (self_key == from_key) || (is_dense(self_key) && is_dense(from_key));
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl(
      "_has_compatible_shallow_copy_type",
      TORCH_FN(_has_compatible_shallow_copy_type));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(
      "_has_compatible_shallow_copy_type",
      TORCH_FN(_has_compatible_shallow_copy_type));
}

} // namespace ops
} // namespace torch_mlu
