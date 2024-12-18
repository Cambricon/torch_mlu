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

#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"
#include "aten/utils/utils.h"

namespace torch_mlu::ops {

static inline void check_device_and_numel(
    const c10::Device& device,
    const int64_t tensor_numel,
    const at::Tensor& tensor) {
  if (!tensor.defined())
    return;
  TORCH_CHECK(tensor.device() == device, "Device need be same.");
  TORCH_CHECK(tensor.numel() == tensor_numel, "Tensor element need be same.");
}

static inline void check_device_and_numel(
    const c10::Device& device,
    const int64_t tensor_numel,
    const std::optional<at::Tensor>& tensor) {
  if (!tensor.has_value())
    return;
  check_device_and_numel(device, tensor_numel, tensor.value());
}

template <
    typename T,
    std::enable_if_t<
        std::is_same_v<T, at::Tensor> ||
            std::is_same_v<T, std::optional<at::Tensor>>,
        int> = 1,
    typename... ARGS>
static inline void check_device_and_numel(
    const c10::Device& device,
    const int64_t tensor_numel,
    const T& tensor,
    ARGS... args) {
  check_device_and_numel(device, tensor_numel, tensor);
  check_device_and_numel(device, tensor_numel, args...);
}

static bool is_high_sqrt_precision() {
  static auto value = []() {
    auto str = std::getenv("TORCH_MLU_SQRT_HIGH_PRECISION");
    if (str != nullptr) {
      std::string value(str);
      if (value == "ON" || value == "on" || value == "1") {
        return true;
      }
    }
    return false;
  }();
  return value;
}

} // end of namespace torch_mlu::ops