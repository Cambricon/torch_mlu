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

#include <c10/util/TypeCast.h>
#include "framework/core/mlu_guard.h"
#include "framework/core/MLUStream.h"
#include "framework/core/memory_allocator.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/utils.h"
#include "framework/hooks/MLUHooks.h"
#include "framework/core/device.h"
#include "framework/core/tensor_impl.h"

namespace torch_mlu {

c10::TensorImpl* getMluTensorImpl(const at::Tensor& tensor) {
  auto tensor_impl = tensor.unsafeGetTensorImpl();
  if (!tensor.defined()) {
    return tensor_impl;
  }
  TORCH_CHECK(
      tensor_impl->device_type() == c10::DeviceType::PrivateUse1,
      "The device type of tensor is not 'mlu'.\n",
      "Please check the python code where the 'result = mlu_model(input)' is called.\n",
      "Please make sure the input.device is 'device(type='mlu', index=0)'.\n\n");

  if (!tensor_impl->external_) {
    tensor_impl->set_external(std::make_unique<MLUTensorImpl>(tensor));
  }
  return tensor_impl;
}

cnnlDataType_t getCnnlType(const c10::TensorImpl* impl) {
  return dynamic_cast<MLUTensorImpl*>(impl->external_.get())->cnnl_data_type_;
}

void setCnnlType(c10::TensorImpl* impl, cnnlDataType_t cnnl_dtype) {
  static_cast<MLUTensorImpl*>(impl->external_.get())->cnnl_data_type_ =
      cnnl_dtype;
}

void copy_to_cpu(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  TORCH_CHECK(
      dst.numel() == src.numel(),
      "Currently only support equal elements D2H copy, but got dst.numel() ",
      dst.numel(),
      " src.numel() ",
      src.numel());

  TORCH_CHECK(
      dst.is_non_overlapping_and_dense() &&
          src.is_non_overlapping_and_dense() &&
          (dst.strides() == src.strides()),
      "cnrtMemcpy don't support stride, dst and src must be non_overlapping_and_dense in D2H.");

  if (src.numel() == 0)
    return;

  torch_mlu::mlu::MLUGuard guard(src.device().index());
  auto src_impl = getMluTensorImpl(src);
  auto src_on_chip_type = cnnlType2ScalarType(getCnnlType(src_impl));
  size_t descriptor_size = src.numel() * c10::elementSize(src_on_chip_type);

  void* output_ptr = nullptr;
  at::Tensor cast_src_cpu_tensor = dst;
  auto stream = getCurrentMLUStream();
  if (src_on_chip_type == dst.dtype()) {
    output_ptr = dst.data_ptr();
    TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
        output_ptr,
        src_impl->mlu_data_ptr(),
        descriptor_size,
        stream.stream(),
        CNRT_MEM_TRANS_DIR_DEV2HOST));
  } else {
    cast_src_cpu_tensor = at::empty_strided(
        src.sizes(),
        src.strides(),
        src.options().dtype(src_on_chip_type).device(at::kCPU));
    cast_src_cpu_tensor._set_conj(src.is_conj());
    cast_src_cpu_tensor._set_neg(src.is_neg());
    output_ptr = cast_src_cpu_tensor.data_ptr();
    TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
        output_ptr,
        src_impl->mlu_data_ptr(),
        descriptor_size,
        stream.stream(),
        CNRT_MEM_TRANS_DIR_DEV2HOST));
  }
  bool is_pinned = isPinnedPtr(output_ptr);
  if (!(non_blocking && is_pinned)) {
    stream.synchronize();
  } else {
    void* ctx = cast_src_cpu_tensor.storage().data_ptr().get_context();
    MLUCachingHostAllocator_recordEvent(output_ptr, ctx, stream);
  }
  if (src_on_chip_type != dst.dtype()) {
    dst.copy_(cast_src_cpu_tensor);
  }
}

// Copy from cpu or to cpu is follow pytorch gpu logic.
// 1 no cpu dytpe convert:
// non_blocking == true and cpu memory is pin memory, aysc process;
// otherwise we need to synchronize stream;
// 2 cpu dytpe convert:
// non_blocking == true, we malloc cpu pin memory, aysc process;
// non_blocking == false, we malloc cpu pageable memory, sync process;
void copy_from_cpu(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  TORCH_CHECK(
      dst.is_non_overlapping_and_dense() &&
          src.is_non_overlapping_and_dense() &&
          (dst.strides() == src.strides()),
      "cnrtMemcpy don't support stride, dst and src must be non_overlapping_and_dense in H2D.");
  TORCH_MLU_CHECK(
      dst.dtype() == src.dtype(),
      "only support src is the same dtype with dst,"
      " but got ",
      src.dtype(),
      " and ",
      dst.dtype());
  if (dst.numel() == 0)
    return;

  if (src.storage().data()) {
    const c10::ScalarType src_type = src.scalar_type();
    at::Tensor cast_cpu_tensor = src;
    // Now mlu not support double and ComplexDouble dtype, so cast to float
    // and ComplexFloat.
    static std::unordered_map<c10::ScalarType, c10::ScalarType>
        cast_dtype_mapping(
            {{c10::ScalarType::Double, c10::ScalarType::Float},
             {c10::ScalarType::ComplexDouble, c10::ScalarType::ComplexFloat}});
    auto it = cast_dtype_mapping.find(src_type);
    if (it != cast_dtype_mapping.end()) {
      TORCH_WARN_ONCE(
          "\033[31m MLU operators don't support 64-bit calculation. ",
          "so the 64 bit data will be forcibly converted to 32-bit for calculation. \033[0m");
      /**** data cast ****/
      const c10::ScalarType target_dtype = it->second;
      // if non_blocking is true, using pin_memory to call asyn memory copy
      // without synchronize. otherwise call asyn memory copy with synchronize.
      auto options =
          src.options().dtype(target_dtype).pinned_memory(non_blocking);
      cast_cpu_tensor = at::empty_like(src, options);
      // call cast_cpu_op to cast dtype.
      if (src_type == c10::ScalarType::Double) {
        cast_cpu_op<float, double>(
            cast_cpu_tensor.data_ptr(), src.data_ptr(), src.numel());
      } else if (src_type == c10::ScalarType::ComplexDouble) {
        cast_cpu_op<float, double>(
            cast_cpu_tensor.data_ptr(), src.data_ptr(), 2 * src.numel());
      } else {
        LOG(ERROR) << "Invalid cast src dtype.";
      }
    }
    /**** cpu2mlu copy ****/
    void* cast_cpu_ptr = cast_cpu_tensor.data_ptr();
    auto dst_impl = getMluTensorImpl(dst);
    void* dst_ptr = dst_impl->mlu_data_ptr();
    auto stream = getCurrentMLUStream(dst.device().index());
    CNRT_CHECK(cnrtMemcpyAsync_V2(
        dst_ptr,
        cast_cpu_ptr,
        cast_cpu_tensor.nbytes(),
        stream.stream(),
        CNRT_MEM_TRANS_DIR_HOST2DEV));
    bool is_pinned = isPinnedPtr(cast_cpu_ptr);
    if (!(non_blocking && is_pinned)) {
      stream.synchronize();
    } else {
      void* ctx = cast_cpu_tensor.storage().data_ptr().get_context();
      MLUCachingHostAllocator_recordEvent(cast_cpu_ptr, ctx, stream);
    }
  }
}

std::vector<int64_t> get_channels_last_strides_1d(
    const at::IntArrayRef& sizes) {
  std::vector<int64_t> strides(sizes.size());
  switch (sizes.size()) {
    // NLC
    case 3:
      strides[1] = 1;
      strides[2] = sizes[1];
      strides[0] = strides[2] * sizes[2];
      return strides;
    // LC
    case 2:
      strides[0] = 1;
      strides[1] = sizes[0];
      return strides;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "ChannelsLast1d doesn't support size ", sizes.size());
  }
}

std::vector<int64_t> get_channels_last_strides(const at::IntArrayRef& sizes) {
  switch (sizes.size()) {
    case 5:
      return c10::get_channels_last_strides_3d(sizes);
    case 4:
      return c10::get_channels_last_strides_2d(sizes);
    case 3:
      return get_channels_last_strides_1d(sizes);
    default:
      TORCH_INTERNAL_ASSERT(
          false, "ChannelsLast doesn't support size ", sizes.size());
  }
}

std::vector<int64_t> get_channels_first_strides(const at::IntArrayRef& sizes) {
  auto dim = sizes.size();
  std::vector<int64_t> strides(dim);
  if (dim > 0) {
    int last_idx = dim - 1;
    strides[last_idx] = 1;
    for (auto i = last_idx - 1; i >= 0; --i) {
      strides[i] = strides[i + 1] * std::max<int64_t>(sizes[i + 1], 1);
    }
  }
  return strides;
}

std::vector<int64_t> get_contiguous_strides(
    const at::IntArrayRef& sizes,
    c10::MemoryFormat memory_format) {
  switch (memory_format) {
    case c10::MemoryFormat::Contiguous:
      return get_channels_first_strides(sizes);
    case c10::MemoryFormat::ChannelsLast:
    case c10::MemoryFormat::ChannelsLast3d:
      return get_channels_last_strides(sizes);
    default:
      TORCH_MLU_CHECK(
          false,
          "get_contiguous_strides doesn't support memory_format ",
          memory_format);
  }
}

bool is_channels_last(const at::Tensor& t) {
  if ((t.dim() < 4) || (t.dim() > 5)) {
    return false;
  } else {
    auto is_channels_last_2d =
        getMluTensorImpl(t)->is_strides_like_channels_last();
    auto is_channels_last_3d =
        getMluTensorImpl(t)->is_strides_like_channels_last_3d();
    return (is_channels_last_2d || is_channels_last_3d);
  }
}

void checkSameMLU(
    at::CheckedFrom c,
    const at::TensorArg& t1,
    const at::TensorArg& t2) {
  TORCH_CHECK(
      t1->device() == t2->device(),
      "Expected tensor for ",
      t1,
      " to have the same device as tensor for ",
      t2,
      "; but first tensor device type ",
      t1->device().type(),
      " and device id ",
      t1->get_device(),
      " does not equal to second tensor ",
      t2->device().type(),
      " and device id ",
      t2->get_device(),
      " (while checking arguments for ",
      c,
      ")");
}

// Check TensorArg t1 and TensorArg t2 with same attribute by using fn function
// ptr.
void checkAllSame(
    at::CheckedFrom c,
    at::ArrayRef<at::TensorArg> tensors,
    void (*fn)(at::CheckedFrom, const at::TensorArg&, const at::TensorArg&)) {
  const at::TensorArg* t0 = nullptr;
  for (auto& t : tensors) {
    if (!t->defined())
      continue;
    if (t0 != nullptr) {
      fn(c, *t0, t);
    } else {
      t0 = &t;
    }
  }
}

void checkAllSameMLU(at::CheckedFrom c, at::ArrayRef<at::TensorArg> tensors) {
  checkAllSame(c, tensors, checkSameMLU);
}

MluFastSetupType switch_memory_format_to_mlu_setup_type(
    const c10::MemoryFormat& memory_format) {
  MluFastSetupType value = MluFastSetupType::CONTIGUOUS;
  switch (memory_format) {
    case c10::MemoryFormat::Contiguous:
      value = MluFastSetupType::CONTIGUOUS;
      break;
    case c10::MemoryFormat::ChannelsLast:
      value = MluFastSetupType::CHANNELS_LAST;
      break;
    case c10::MemoryFormat::ChannelsLast3d:
      value = MluFastSetupType::CHANNELS_LAST_3D;
      break;
    default:
      value = MluFastSetupType::NONE;
      break;
  }
  return value;
}

// Compute tensors setup type based on tensor list,
// Based on priority in MluFastSetupType class, we check each tensor's memory
// status, and return corresponding MluFastSetupType.
MluFastSetupType compute_tensors_setup_type(
    const std::vector<at::TensorBase>& tensor_list) {
  if (tensor_list.size() == 0) {
    return MluFastSetupType::NONE;
  }
  bool is_contiguous = true;
  bool is_channels_last = true;
  bool is_channels_last3d = true;
  bool is_non_overlapping_and_dense = true;
  for (const auto& item : tensor_list) {
    TORCH_CHECK(item.defined(), "Tensor is not defined.");
    is_contiguous &= item.is_contiguous(at::MemoryFormat::Contiguous);
    is_channels_last &= item.is_contiguous(at::MemoryFormat::ChannelsLast);
    is_channels_last3d &= item.is_contiguous(at::MemoryFormat::ChannelsLast3d);
    is_non_overlapping_and_dense &= item.is_non_overlapping_and_dense();
  }
  // This leads to ambiguous cases (NC11) to be always treated as
  // Channels_last or Channels_last_3d.
  if (is_channels_last) {
    return MluFastSetupType::CHANNELS_LAST;
  }
  if (is_channels_last3d) {
    return MluFastSetupType::CHANNELS_LAST_3D;
  }
  if (is_contiguous) {
    return MluFastSetupType::CONTIGUOUS;
  }
  if (is_non_overlapping_and_dense) {
    auto sizes = tensor_list[0].sizes();
    auto strides = tensor_list[0].strides();
    for (const auto& item : tensor_list) {
      if (sizes != item.sizes() || strides != item.strides()) {
        // when all input tensors are NON_OVERLAPPING_DENSE,
        // but size or stride is different. Using first tensor memory format
        // to get mlu setup type.
        // Here is a little with op memory format config, if return None here,
        // it will fallback to channel last in dim value equal to 4 or 5.
        // So now keep same behavior with before, using first tensor memory
        // format.
        return switch_memory_format_to_mlu_setup_type(
            tensor_list[0].suggest_memory_format());
      }
    }
    return MluFastSetupType::NON_OVERLAPPING_DENSE;
  }
  // cnnl kernel not support overlapping or not dense tensor, so finally return
  // NONE to call cnnl_contiguous to get channels_first contiguous tensors.
  return MluFastSetupType::NONE;
}

// Convert 64-bit ScalarType to 32-bit ScalarType.
at::ScalarType get_mlu_scalar_type(const at::ScalarType& scalar_type) {
  switch (scalar_type) {
    case at::ScalarType::Double:
      return at::ScalarType::Float;
    case at::ScalarType::ComplexDouble:
      return at::ScalarType::ComplexFloat;
    default:
      return scalar_type;
  }
}

// Parameters:
// common_dtype is computed by promoteTypes in pytorch side.
// support_types is cnnl kernel supported types.
// convert_dtype is a flag to control whether implicit convert common type to
// cnnl dtype. Return: compute type is TORCH_MLU promoteTypes based on support
// types.
// 1. Get promote compute types based on cnnl kernel supported types and common
// type,
// 2. if implicit convert is supported, get best compute dtype in promote
// compute types.
at::ScalarType get_torch_mlu_promote_types(
    const at::ScalarType& common_dtype_,
    const std::vector<at::ScalarType>& support_types,
    const std::string& op_name,
    bool convert_dtype) {
  // cause cnnl kernel not support tensor dtype double/complexDouble,
  // so convert common dtype to float or int or complex float.
  at::ScalarType compute_dtype_ = at::ScalarType::Float;
  TORCH_CHECK(support_types.size() > 0, "support types is empty.");
  // calculate compute dtype based on common dtype and op compute dtype.
  auto it =
      std::find(support_types.begin(), support_types.end(), common_dtype_);
  if (it != support_types.end()) {
    compute_dtype_ = common_dtype_;
    return compute_dtype_;
  }
  TORCH_CHECK(
      convert_dtype,
      "MLU ",
      op_name,
      " don't support tensor dtype ",
      common_dtype_,
      ".");
  // based on common dtype and compute dtype, get a better dtype.
  std::vector<at::ScalarType> v_promote_type;
  auto common_dtype_size = c10::scalarTypeToTypeMeta(common_dtype_).itemsize();
  for (const auto& type : support_types) {
    // call native promote_type to get a better type between in
    // cnnl op support dtype and common_dtype, and store promote one
    // in promote_type_vec, which width of promote type must larger or
    // equal than common_dtype. Also promote type one need be defined
    // and in cnnl op support list.
    auto ret = c10::promoteTypes(common_dtype_, type);
    auto ret_dtype_size = c10::scalarTypeToTypeMeta(ret).itemsize();
    // Add dtype width compare. If promote dtype width is smaller than
    // common dtype width, we need to skip it.
    if (ret != at::ScalarType::Undefined && ret != common_dtype_ &&
        ret_dtype_size >= common_dtype_size) {
      // Check whether promote type is contained in support_types.
      auto it = std::find(support_types.begin(), support_types.end(), ret);
      if (it != support_types.end()) {
        v_promote_type.emplace_back(ret);
      }
    }
  }

  TORCH_CHECK(
      v_promote_type.size() > 0,
      "MLU ",
      op_name,
      "Can't get a promote dtype when common dtype is ",
      common_dtype_);
  // Choose the smallest width of compute dtype for accelerateing op compute.
  // Also will using more smaller device memory.
  compute_dtype_ = v_promote_type[0];
  for (const auto& type : v_promote_type) {
    if (static_cast<int8_t>(compute_dtype_) > static_cast<int8_t>(type)) {
      compute_dtype_ = type;
    }
  }
  TORCH_CHECK(
      at::canCast(common_dtype_, compute_dtype_),
      "Can't cast common dtype ",
      common_dtype_,
      " to cnnl op compute type.");
  return compute_dtype_;
}

bool compute_channels_last_contiguous_2d(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  const int dim = size.size();
  TORCH_MLU_CHECK(dim == stride.size(), "size and stride size need be equal.");
  switch (dim) {
    case 4: {
      int64_t expected = 1;
      for (auto& d : {1, 3, 2, 0}) {
        if (size[d] != 1) {
          if (stride[d] != expected) {
            return false;
          }
          expected *= size[d];
        }
      }
      return true;
    }
    case 3:
      // TODO(shangang): dim == 3 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

bool compute_channels_last_contiguous_3d(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  const int dim = size.size();
  TORCH_MLU_CHECK(dim == stride.size(), "size and stride size need be equal.");
  switch (dim) {
    case 5: {
      int64_t expected = 1;
      for (auto& d : {1, 4, 3, 2, 0}) {
        if (size[d] != 1) {
          if (stride[d] != expected) {
            return false;
          }
          expected *= size[d];
        }
      }
      return true;
    }
    case 4:
      // TODO(shangang): dim == 4 case will be enabled once it is fully tested
      return false;
    default:
      return false;
  }
}

std::vector<int64_t> compute_permute_dims_to_contiguous(
    const at::Tensor& tensor) {
  TORCH_MLU_CHECK(
      tensor.is_non_overlapping_and_dense(),
      "Tensor must be non-overlapping and dense.")
  TORCH_MLU_CHECK(
      tensor.numel() > 0,
      "zero-element tensor is always contiguous, so not support.");
  const int nDim = tensor.dim();
  std::vector<int64_t> perm;
  perm.resize(nDim);
  std::iota(perm.begin(), perm.end(), 0);
  // Fast path if tensor is contiguous with suggest memory format.
  auto memory_format = tensor.suggest_memory_format();
  if (tensor.is_contiguous(memory_format)) {
    switch (memory_format) {
      case c10::MemoryFormat::Contiguous:
        return perm;
      case c10::MemoryFormat::ChannelsLast:
        return std::vector<int64_t>{0, 2, 3, 1};
      case c10::MemoryFormat::ChannelsLast3d:
        return std::vector<int64_t>{0, 2, 3, 4, 1};
      default:
        TORCH_MLU_CHECK(false, "Not support memory format ", memory_format);
    }
  }
  // Slow path if tensor is not contiguous with suggest memory format.
  const auto& size = tensor.sizes();
  const auto& stride = tensor.strides();
  auto should_swap = [&size, &stride](int64_t i, int64_t j) -> bool {
    // Stride can be anything when size dim value is 1.
    // For most situation, stride is equal to ahead dim stride value when size
    // dim value is 1. So put size dim value is 1 behead of size dim value is
    // not 1 when stride is equal.
    if (stride[i] == stride[j]) {
      return size[i] < 2 && size[j] >= 2 ? true : false;
    }
    return stride[i] < stride[j];
  };
  // insert sort by stride and size value
  for (int i = 1; i < nDim; ++i) {
    for (int j = i; j > 0 && should_swap(perm[j - 1], perm[j]); --j) {
      std::swap(perm[j - 1], perm[j]);
    }
  }
  return perm;
}

// More details in .h
bool geometry_is_cl_contiguous(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride) {
  const int dim = size.size();
  // CL contiguous tensor dim need be equal to 4 or 5.
  if (dim != 4 && dim != 5) {
    return false;
  }
  return torch_mlu::compute_channels_last_contiguous_2d(size, stride) ||
      torch_mlu::compute_channels_last_contiguous_3d(size, stride);
}

// More details in .h
bool is_geometry_contiguous(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride) {
  if (at::geometry_is_contiguous(size, stride) ||
      torch_mlu::geometry_is_cl_contiguous(size, stride)) {
    return true;
  }
  return false;
}

// More details in .h
bool check_device(at::ArrayRef<at::Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  at::Device curDevice = at::Device(at::kPrivateUse1, current_device());
  for (const at::Tensor& t : ts) {
    if (t.device() != curDevice)
      return false;
  }
  return true;
}

bool is_same_strides(
    const at::IntArrayRef& size,
    const at::IntArrayRef& stride,
    const at::IntArrayRef& default_stride) {
  const int dim = size.size();
  TORCH_MLU_CHECK(
      dim == stride.size() && dim == default_stride.size(),
      "size and stride must have same size");
  // check if all strides are same
  for (int i = 0; i < dim; ++i) {
    if (size[i] == 1)
      continue;
    if (stride[i] != default_stride[i]) {
      return false;
    }
  }
  return true;
}

} // namespace torch_mlu
