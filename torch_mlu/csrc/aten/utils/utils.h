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

#include <algorithm>
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "utils/Export.h"
#include "framework/hooks/MLUHooks.h"

namespace torch_mlu {
cnnlTensorLayout_t suggest_cnnl_layout(const at::Tensor& input);

std::vector<int64_t> get_permute_back_order(const at::Tensor& input);

at::Tensor cast_long_to_int_if_needed(
    const at::Tensor& input,
    const bool need_check = true);

at::Tensor create_int_tensor_if_needed(const at::Tensor& output);

void cast_int_to_long_if_needed(
    const at::Tensor& output,
    const at::Tensor& out);

bool is_same_format_tensor(const at::TensorList& tensors);

at::Tensor svd_backward(
    const std::vector<torch::autograd::Variable>& grads,
    const at::Tensor& self,
    bool some,
    bool compute_uv,
    const at::Tensor& raw_u,
    const at::Tensor& sigma,
    const at::Tensor& raw_v);

at::Tensor unsqueeze_multiple(
    const at::Tensor& t,
    at::IntArrayRef dim,
    size_t n_dims);

c10::MemoryFormat switch_tensors_suggest_memory_format(
    const std::vector<at::Tensor>& tensor_list);

void copy_to_cpu(at::Tensor& dst, const at::Tensor& src, bool non_blocking);

void copy_from_cpu(
    at::Tensor& dst,
    const at::Tensor& src,
    bool non_blocking = false);

bool is_channels_last(const at::Tensor& t);

std::vector<int64_t> get_channels_last_strides_1d(const at::IntArrayRef& sizes);

std::vector<int64_t> get_channels_last_strides(const at::IntArrayRef& sizes);

std::vector<int64_t> get_channels_first_strides(const at::IntArrayRef& sizes);

std::vector<int64_t> get_contiguous_strides(
    const at::IntArrayRef& sizes,
    c10::MemoryFormat memory_format = c10::MemoryFormat::Contiguous);

void checkAllSameMLU(at::CheckedFrom c, at::ArrayRef<at::TensorArg> tensors);

void checkSameMLU(
    at::CheckedFrom c,
    const at::TensorArg& t1,
    const at::TensorArg& t2);

// torch tensor provides int64_t type of shape and stride,
// but cnnl descriptor requires type int32.
// use this function to ensure safe CAST, or report an error.
template <typename DST_T, typename SRC_T>
inline std::vector<DST_T> checkUpperBoundAndCastTo(
    const std::vector<SRC_T>& input) {
  std::vector<DST_T> output;
  output.reserve(input.size());
  for (const auto& val : input) {
    if (val > std::numeric_limits<DST_T>::max()) {
      TORCH_CHECK(
          false,
          "Requires dim size not greater than ",
          std::numeric_limits<DST_T>::max(),
          ". But got ",
          val,
          ".");
    }
    output.push_back(static_cast<DST_T>(val));
  }
  return output;
}

inline int getTensorDevice(at::Tensor tensor) {
  int device_index = current_device();
  if (tensor.device().type() == at::kPrivateUse1) {
    device_index = tensor.device().index();
  }
  return device_index;
}

inline int getTensorDevice(at::TensorList list) {
  int device_index = current_device();
  for (auto& t : list) {
    if (t.device().type() == at::kPrivateUse1) {
      device_index = t.device().index();
      break;
    }
  }
  return device_index;
}

inline int getTensorDevice(at::TensorOptions options) {
  int device_index = current_device();
  if (options.device().type() == at::kPrivateUse1) {
    device_index = options.device_index();
  }
  return device_index;
}

template <typename T>
struct value_type {
  typedef T type;
};

template <typename T>
struct value_type<c10::complex<T>> {
  typedef T type;
};

template <typename T>
inline T numeric_min() {
  return std::numeric_limits<typename value_type<T>::type>::lowest();
}

template <typename T>
inline T numeric_max() {
  return std::numeric_limits<typename value_type<T>::type>::max();
}

template <class DST, class SRC>
inline constexpr bool is_downcast() {
  return sizeof(SRC) > sizeof(DST);
}

template <typename DST, typename SRC>
class downcast_range_checker {
 public:
  static bool is_valid(SRC& src) {
    return (src <= numeric_max<DST>()) && (src >= numeric_min<DST>());
  }
};

template <typename DST, typename SRC>
class downcast_range_checker<c10::complex<DST>, c10::complex<SRC>> {
 public:
  static bool is_valid(c10::complex<SRC>& src) {
    return (src.real() <= numeric_max<DST>()) &&
        (src.real() >= numeric_min<DST>()) &&
        (src.imag() <= numeric_max<DST>()) &&
        (src.imag() >= numeric_min<DST>());
  }
};

// in this overload, std::true_type means
// downcasting from high precision to low precision
// limits check is needed
template <typename DST, typename SRC>
inline void cast_cpu_op_imp(
    DST* pDst,
    SRC* pSrc,
    int64_t numel,
    std::true_type) {
  for (size_t i = 0; i < numel; ++i) {
    SRC& raw = *(pSrc + i);
    using Checker = downcast_range_checker<DST, SRC>;
    TORCH_CHECK(
        // raw <= std::numeric_limits<DST>::max() && raw >= numeric_min<DST>(),
        Checker::is_valid(raw),
        "datacast fail! expected smaller than ",
        // std::numeric_limits<DST>::max(),
        numeric_max<DST>(),
        " and greater than ",
        numeric_min<DST>(),
        "but got ",
        raw);
    *(pDst + i) = raw;
  }
}

// for efficiency sake, no check is performed in this overload
template <typename DST, typename SRC>
inline void cast_cpu_op_imp(
    DST* pDst,
    SRC* pSrc,
    int64_t numel,
    std::false_type) {
  for (size_t i = 0; i < numel; ++i) {
    SRC raw = *(pSrc + i);
    *(pDst + i) = raw;
  }
}

template <typename DST, typename SRC>
inline void cast_cpu_op(void* dst_cpu_ptr, void* src_cpu_ptr, int64_t numel) {
  SRC* pSrc = static_cast<SRC*>(src_cpu_ptr);
  DST* pDst = static_cast<DST*>(dst_cpu_ptr);
  cast_cpu_op_imp(
      pDst,
      pSrc,
      numel,
      typename std::conditional<
          is_downcast<DST, SRC>(),
          std::true_type,
          std::false_type>::type());
}

template <typename T>
inline bool isPinnedPtr(T* ptr) {
  at::OptionalDeviceGuard device_guard;
  auto shared_ctx_device_index = torch_mlu::getDeviceIndexWithSharedContext();
  if (shared_ctx_device_index.has_value()) {
    device_guard.reset_device(
        at::Device(at::Device(at::kPrivateUse1, *shared_ctx_device_index)));
  }
  cnrtPointerAttributes_t attr;
  cnrtRet_t status = cnrtPointerGetAttributes(&attr, const_cast<void*>(ptr));
  if (status == cnrtErrorArgsInvalid) {
    (void)cnrtGetLastError(); // clear cnrt error
    return false;
  }
  TORCH_CNRT_CHECK(status);
  return attr.type == cnrtMemTypeHost;
}

enum class MluFastSetupType : uint8_t {
  NONE,
  CONTIGUOUS,
  CHANNELS_LAST,
  CHANNELS_LAST_3D,
  NON_OVERLAPPING_DENSE
};

MluFastSetupType switch_memory_format_to_mlu_setup_type(
    const c10::MemoryFormat& setup_type);

MluFastSetupType compute_tensors_setup_type(
    const std::vector<at::TensorBase>& tensor_list);

at::ScalarType get_torch_mlu_promote_types(
    const at::ScalarType& common_dtype_,
    const std::vector<at::ScalarType>& support_types,
    const std::string& op_name,
    bool convert_dtype = false);

// Copy from c10/core/TensorImpl.cpp
// Tensor impl private func.
bool compute_channels_last_contiguous_2d(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride);

bool compute_channels_last_contiguous_3d(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride);

// If tensor is satisfied with is_non_overlapping_and_dense, and compute permute
// dims from contiguous tensor to this is_non_overlapping_and_dense tensor.
std::vector<int64_t> compute_permute_dims_to_contiguous(
    const at::Tensor& tensor);

// Compare to pytorch function geometry_is_contiguous, is_geometry_cl_contiguous
// is for channels last and channels last 3d contigous.
bool geometry_is_cl_contiguous(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride);

// Return if the tensor geometry represented by `sizes` and `strides` is
// contiguous Although we cache is_contiguous in tensor now, this is till useful
// because it allows checking if a particular geometry is contiguous without
// explicitly constructing a tensor, e.g., when you want to choose a kernel
// strategy based on whether a subgeometry is contiguous.(Support channels last
// and channels last 3d contigous)
bool is_geometry_contiguous(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride);

// Check if every tensor in a list of tensors matches the current
// device.
bool check_device(at::ArrayRef<at::Tensor> ts);

// Convert 64-bit ScalarType to 32-bit ScalarType.
at::ScalarType get_mlu_scalar_type(const at::ScalarType& scalar_type);

// check scalar type is same with default scalar type without 64bit ScalarType.
template <
    typename T,
    typename = std::enable_if_t<
        std::is_same<typename std::decay<T>::type, at::ScalarType>::value,
        bool>>
inline bool is_same_types(T&& default_type, T&& type) {
  if (default_type == type || default_type == get_mlu_scalar_type(type) ||
      get_mlu_scalar_type(default_type) == type) {
    return true;
  }
  return false;
}

template <typename... ARGS>
inline bool is_same_types(ARGS&&... args) {
  // check if all types are same
  constexpr int num = sizeof...(args);
  static_assert(num >= 2, "is_same_types requires at least 2 arguments");
  std::array<at::ScalarType, num> types{args...};
  at::ScalarType type = get_mlu_scalar_type(types[0]);
  for (int i = 1; i < num; ++i) {
    if (!is_same_types(type, types[i])) {
      return false;
    }
  }
  return true;
}

// check strides is same with default strides, and except dim size value
// is 1.
bool is_same_strides(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& default_stride);

} // namespace torch_mlu
