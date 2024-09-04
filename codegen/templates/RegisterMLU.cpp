/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/pytorch/pytorch/graphs/contributors
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
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

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/Operators.h>
#include <torch/library.h>

#include "aten/MLUFallback.h"
#include "aten/generated/MLUFunctions.h"
#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnfft_plan_cache.h"
#include "aten/operators/mluop/mluop_kernel.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {

static bool enable_mlu_fallback = getFailFallbackEnabledEnvVar();

at::Tensor create_out(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  if (strides.empty()) {
    return torch_mlu::mlu::empty(sizes, options);
  } else {
    return torch_mlu::mlu::empty_strided(sizes, strides, options);
  }
}

// Add a new interface for TensorIterator OP, CATCH will modify
// tensor stride to get a contiguous tensor.
void create_out_or_resize(
    at::Tensor& out,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  if (!out.defined()) {
    out = create_out(sizes, strides, options);
  } else {
    TORCH_CHECK(sizes == out.sizes(), "out.sizes() and sizes must be "
    "equal, but got ", out.sizes(), " and ", sizes);
    if (!strides.empty() && !strides.equals(out.strides())) {
      out.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, strides);
    }
  }
}

void resize_out(
    const at::Tensor& out,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  TORCH_CHECK(
      options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      options.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      options.device() == out.device(),
      "Expected out tensor to have device ",
      options.device(),
      ", but got ",
      out.device(),
      " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  // MLU side don't check the resized status, cause CATCH always need contiguous
  // output tensor. Except when output tensor is nondense, resized should be checked,
  // only when resized is true, output tensor can be restrided.
  if (!strides.empty()) {
    TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
    // TODO: avoid the redispatch here
    out.as_strided_(sizes, strides);
  } else if (resized && options.memory_format_opt().has_value()) {
    out.unsafeGetTensorImpl()->empty_tensor_restride(
        *options.memory_format_opt());
  }
}
void check_inplace(
    const at::Tensor& self,
    at::IntArrayRef sizes,
    const at::TensorOptions& options) {
  // These checks are needed on those operators that:
  //   1) don't use 'TensorIterator' (e.g. 'addmm' and 'baddbmm')
  //   2) have particular typing rules (e.g. 'cumsum' and 'cumprod')
  // For other operators (e.g. 'add'), 'TensorIterator' already checks
  // these things separately.
  TORCH_CHECK(
      options.dtype() == self.dtype(),
      "Bad in-place call: ",
      "input tensor dtype ",
      self.dtype(),
      " and output tensor dtype ",
      options.dtype(),
      " should match");
  TORCH_CHECK(
      options.device() == self.device(),
      "Bad in-place call: ",
      "input tensor device ",
      self.device(),
      " and output tensor device ",
      options.device(),
      " should match");
  TORCH_CHECK(
      sizes == self.sizes(),
      "Bad in-place call: ",
      "input tensor size ",
      self.sizes(),
      " and output tensor size ",
      sizes,
      " should match");
}
c10::optional<at::Tensor> maybe_create_proxy(
    const at::Tensor& out,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  if (out.strides() != strides) {
    return torch_mlu::mlu::empty_strided(sizes, strides, options);
  }
  return c10::nullopt;
}

// op has fallback fn
template <typename Return, typename F1, typename F2, typename... Args>
static Return op_call(F1 impl_call, F2 fallback_call, Args&&... args) {
  if (enable_mlu_fallback) {
    try {
      return impl_call(std::forward<Args>(args)...);
    } catch (std::exception& e) {
      CNLOG(INFO) << e.what();
      return fallback_call(std::forward<Args>(args)...);
    }
  }
  return impl_call(std::forward<Args>(args)...);
}

// op has no fallback fn
template <typename Return, typename F, typename... Args>
static Return op_call2(F impl_call, Args&&... args) {
  return impl_call(std::forward<Args>(args)...);
}

namespace {

namespace {
int64_t wrapper__cnfft_get_plan_cache_size(at::DeviceIndex device_index) {
  // No device check
  // DeviceGuard omitted
  return torch_mlu::ops::detail::cnfft_get_plan_cache_size_impl(device_index);
}
}  // anonymous namespace
namespace {
int64_t wrapper__cnfft_get_plan_cache_max_size(at::DeviceIndex device_index) {
  // No device check
  // DeviceGuard omitted
  return torch_mlu::ops::detail::cnfft_get_plan_cache_max_size_impl(device_index);
}
}  // anonymous namespace
namespace {
void wrapper__cnfft_set_plan_cache_max_size(at::DeviceIndex device_index, int64_t max_size) {
  // No device check
  // DeviceGuard omitted
  return torch_mlu::ops::detail::cnfft_set_plan_cache_max_size_impl(device_index, max_size);
}
}  // anonymous namespace
namespace {
void wrapper__cnfft_clear_plan_cache(at::DeviceIndex device_index) {
  // No device check
  // DeviceGuard omitted
  return torch_mlu::ops::detail::cnfft_clear_plan_cache_impl(device_index);
}
}  // anonymous namespace

// wrapper definition
${dispatch_anonymous_definitions}

// pytorch aten op registration
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  ${dispatch_aten_registrations}
}

// pytorch aten op registration that need to write
// custom autograd impl
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  ${dispatch_aten_autograd_registrations}
}

// torchvision op registration
TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
  ${dispatch_vision_registrations}
}

// torchaudio op registration
TORCH_LIBRARY_IMPL(torchaudio, PrivateUse1, m) {
  ${dispatch_audio_registrations}
}

// MLU custom op registration
TORCH_LIBRARY(torch_mlu, m) {
  ${custom_schema_registrations}
  m.def("_cnfft_get_plan_cache_size(DeviceIndex device_index) -> int", tags_0);
  m.def("_cnfft_get_plan_cache_max_size(DeviceIndex device_index) -> int", tags_0);
  m.def("_cnfft_set_plan_cache_max_size(DeviceIndex device_index, int max_size) -> ()", tags_0);
  m.def("_cnfft_clear_plan_cache(DeviceIndex device_index) -> ()", tags_0);
}

// MLU custom ops that need create backward graph node
TORCH_LIBRARY_IMPL(torch_mlu, AutogradPrivateUse1, m) {
  ${dispatch_custom_autograd_registrations}
}

// MLU custom ops that do not need backward node
TORCH_LIBRARY_IMPL(torch_mlu, PrivateUse1, m) {
  ${dispatch_custom_registrations}
}

TORCH_LIBRARY_IMPL(torch_mlu, CompositeImplicitAutograd, m) {
  m.impl("_cnfft_get_plan_cache_size", TORCH_FN(wrapper__cnfft_get_plan_cache_size));
  m.impl("_cnfft_get_plan_cache_max_size", TORCH_FN(wrapper__cnfft_get_plan_cache_max_size));
  m.impl("_cnfft_set_plan_cache_max_size", TORCH_FN(wrapper__cnfft_set_plan_cache_max_size));
  m.impl("_cnfft_clear_plan_cache", TORCH_FN(wrapper__cnfft_clear_plan_cache));
}

}  // anonymous namespace

namespace mlu {

${dispatch_namespaced_definitions}

int64_t _cnfft_get_plan_cache_size(at::DeviceIndex device_index) {
  return wrapper__cnfft_get_plan_cache_size(device_index);
}

int64_t _cnfft_get_plan_cache_max_size(at::DeviceIndex device_index) {
  return wrapper__cnfft_get_plan_cache_max_size(device_index);
}

void _cnfft_set_plan_cache_max_size(at::DeviceIndex device_index, int64_t max_size) {
  return wrapper__cnfft_set_plan_cache_max_size(device_index, max_size);
}

void _cnfft_clear_plan_cache(at::DeviceIndex device_index) {
  return wrapper__cnfft_clear_plan_cache(device_index);
}

}  // mlu

}  // namespace torch_mlu
