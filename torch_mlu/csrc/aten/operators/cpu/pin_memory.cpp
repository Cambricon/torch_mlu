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

#include <ATen/ATen.h>
#include <torch/library.h>

#include "aten/operators/cpu/cpu_kernel.h"
#include "aten/utils/ignore_warning_handler.h"
namespace torch_mlu {
bool is_pinned(const at::Tensor& self, std::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }
  c10::DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(
      c10::nullopt,
      self.layout(),
      device.value_or(c10::DeviceType::PrivateUse1)));
  return at::_ops::is_pinned::redispatch(_dk, self, device);
}

at::Tensor _pin_memory(
    const at::Tensor& self,
    std::optional<at::Device> device) {
  TORCH_CHECK(
      self.device().is_cpu(),
      "cannot pin '",
      self.toString(),
      "' only dense CPU tensors can be pinned");
  c10::DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(
      c10::nullopt,
      self.layout(),
      device.value_or(c10::DeviceType::PrivateUse1)));
  return at::_ops::_pin_memory::redispatch(_dk, self, device);
}

#define KERNEL_REGISTER                                                       \
  TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {                                \
    m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(is_pinned));     \
    m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(_pin_memory)); \
  }

WITH_IGNORE_WARNING_OVERRIDE_OPERATOR(true, KERNEL_REGISTER)

} // namespace torch_mlu
