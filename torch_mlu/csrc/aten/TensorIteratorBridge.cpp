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

#include "ATen/native/Resize.h"
#include "aten/utils/cnnl_util.h"
#include "aten/operators/cnnl/cnnlOpParams.h"
#include "aten/TensorIteratorBridge.h"
#include "aten/generated/MLUFunctions.h"

namespace torch_mlu {

// This function is a little different with pytorch original function.
// In pytorch side, the function is defined in TensorIterator.h, and the
// function is using op.target_dtype to create a TensorOptions. In Catch side,
// we need to create internal tensor with compute dtype and create a output
// tensor with oprand target dtype.
at::TensorOptions TensorIteratorBridge::original_options(
    const at::OperandInfo& op,
    const at::ScalarType& scalar_dtype) {
  if (op.original_tensor_base().defined()) {
    return op.original_tensor_base()
        .options()
        .dtype(scalar_dtype)
        .device(op.device);
  } else {
    return at::TensorOptions(scalar_dtype).device(op.device);
  }
}

// Modify op tensor info and keep origin tensor info in op.original_tensor_base.
// Always keep pytorch original tensor in op.original_tensor, and keep
// internal tensor in op.tensor.
// 1) op.original_tensor is undefined, exchange tensor and original_tensor,
// and update operand original tensor and operand tensor.
// 2) Otherwise just cast tensor to target dtype and update operand tensor.
void TensorIteratorBridge::update_operand_tensor_info(
    at::OperandInfo& op,
    at::Tensor&& tensor) {
  if (!op.original_tensor_base().defined()) {
    op.exchange_tensor(
        c10::MaybeOwned<at::TensorBase>::owned(std::move(tensor)));
  } else {
    op.tensor(c10::MaybeOwned<at::TensorBase>::owned(std::move(tensor)));
  }
  op.current_dtype = op.tensor_base().scalar_type();
}

// Check cpu scalar tensor.
bool TensorIteratorBridge::is_cpu_scalar(const at::OperandInfo& op) {
  return op.tensor().numel() == 1 &&
      op.tensor().device().type() == at::DeviceType::CPU;
}

// Compare current dtype and target dtype, if not equal, CATCH need cast tensor
// to target dtype for cnnl kernel. Now MLU tensor don't support 64bit dtype, if
// tensor scalar type is 64bit, the real memory type is already 32bit, so no
// need to call cast op.
bool TensorIteratorBridge::is_same_current_and_target_dtype(
    const at::OperandInfo& op) {
  return is_same_types(op.current_dtype, op.target_dtype);
}

// Check strides is same with common strides or not.
// Dim value is 1, don't need to check.
// https://github.com/pytorch/pytorch/blob/release/1.13/c10/core/TensorImpl.cpp#L283
bool TensorIteratorBridge::is_same_stride(const at::OperandInfo& op) {
  const auto& sizes = op.tensor_base().sizes();
  TORCH_CHECK(
      sizes.size() == this->ndim_, "Strides or sizes is not equal with ndim.");
  return is_same_strides(sizes, op.tensor_base().strides(), this->strides_);
}

// when operator has not inputs tensor, we only need to guarantee two
// conditions:
// 1. output tensor's memory is dense and no overlapping.
// 2. output tensor's dtype can be supported by cnnl kernel.
bool TensorIteratorBridge::nullary_input(
    at::TensorIteratorBase& iter,
    const CnnlOpParams& params,
    const std::string& op_name) {
  auto noutputs = iter.noutputs();
  if (iter.ntensors() != noutputs)
    return false;
  TORCH_CHECK(
      noutputs == 1,
      "Currently TensorIteratorBridge only support single output if nullary op.");
  // nullary op common_dtype() is undefined, and get from output tensor dtype.
  auto common_dtype = iter.dtype();
  // Check whether fix output dtype.
  this->compute_dtype_ = this->fix_output_dtype_;
  if (this->fix_output_dtype_ == at::ScalarType::Undefined) {
    this->compute_dtype_ = get_catch_promote_types(
        common_dtype,
        params.support_types_,
        op_name,
        params.allow_implicit_type_convert_);
  }
  // Only one output tensor, so we can use the first operand to represent the
  // output.
  auto& op = iter.operand(0);
  op.target_dtype = this->compute_dtype_;
  const bool is_same_dtype = is_same_current_and_target_dtype(op);
  // Support strided memory.
  if (params.allow_strided_memory_) {
    if (!is_same_dtype) {
      // CNNL Cast op is not support stride, so we will get a contiguous tensor.
      op.exchange_tensor(c10::MaybeOwned<at::TensorBase>::owned(
          op.tensor().to(op.target_dtype)));
      op.current_dtype = op.target_dtype;
    }
    return true;
  }
  // Don't need to convert common_dtype to 32-bit dtype, common dtype is return
  // dtype, which is decided in TensorIteratorBase.
  if (!op.tensor_base().is_non_overlapping_and_dense() || !is_same_dtype) {
    // Nullary op will_resize is always false, so don't need to check
    // will_resize. target_dtype tensor is using for internal cnnl compute.
    op.exchange_tensor(
        c10::MaybeOwned<at::TensorBase>::owned(torch_mlu::mlu::empty(
            op.tensor_base().sizes(), original_options(op, op.target_dtype))));
    op.current_dtype = op.target_dtype;
  }
  return true;
}

/**
 * Note [CATCH dtype promote]
 * Using operand current dtype and target dtype to represent tensor status.
 * 1) current dtype is mean the pytorch output tensor dtype;
 * 2) target dtype is mean the cnnl compute dtype.
 *
 * The basic policy of dtype assign is:
 * 1) If mixed input types is supported by op, then check input tensor types
 * with mix type list. 2) If mixed input types matched and output dtype is
 * matched exactly, just using output dtype; 3) If mixed input types matched and
 * common dtype is matched exactly, using common dtype and set output operand
 * target dtype to common dtype; 4) If mixed input types matched and
 * output/common dtype are not matched, then fallback to compute dtype. 5) Using
 * common dtype to get cnnl compute dtype, and set target dtype of each operand
 * to compute dtype.
 *
 * The basic policy of output tensor dtype is fellow the pytorch rule, and the
 * output dytpe must same with gpu side.
 *
 */

/**
 * Note [CATCH mixed types]
 * ~~~~~~~~~~~~~~~~
 * CATCH mixed types is decided by CNNL op kernel, and the mixed types list is
 * stored in opParamsMap.
 *
 * Mixed type data format is a unordered map of vector, and the first elements
 * are input types, the last element is output type. For example: 'half + float
 * = half' and 'half + float = float', the mixed types list is {{"hash_value1":
 * {half, float, half}},
 * {"hash_value1": {half, float, float}}}.
 *
 */

void TensorIteratorBridge::compute_types(
    at::TensorIteratorBase& iter,
    const CnnlOpParams& params,
    const std::string& op_name) {
  // Do nothing when op don't need to check types.
  if (params.close_types_check_ == true)
    return;
  auto common_dtype = iter.common_dtype();
  const int n_outputs = iter.noutputs();
  const int n_inputs = iter.ninputs();
  if (params.isSupportMixedInputTypes() && n_outputs == 1 && n_inputs >= 2) {
    // Check input tensor types.
    auto first_input_dtype = iter.dtype(n_outputs);
    bool is_different_input_types = false;
    for (int i = (n_outputs + 1); i < (n_outputs + n_inputs); ++i) {
      if (!is_same_types(first_input_dtype, iter.dtype(i))) {
        is_different_input_types = true;
        break;
      }
    }
    if (is_different_input_types == true) {
      auto getOutputTypeWithMixedInputTypes =
          [&params](const std::vector<at::ScalarType>& vec_types) -> bool {
        const int64_t hash_value =
            getScalarTypeListHashWithOutputType(vec_types);
        auto it = params.input_mixed_types_map_.find(hash_value);
        // 1) If mixed types found, check output dtype and set operand target
        // dtype. 2) if mixed types not found, fallback to common compute type
        // promote.
        if (it != params.input_mixed_types_map_.end()) {
          return true;
        }
        return false;
      };
      // 1) Using output tensor type to check with mixed types list;
      std::vector<at::ScalarType> item_types;
      item_types.reserve(n_inputs + n_outputs);
      // first add input types, then add output types.
      for (int i = 0; i < n_inputs; ++i) {
        item_types.emplace_back(iter.dtype(n_outputs + i));
      }
      for (int i = 0; i < n_outputs; ++i) {
        item_types.emplace_back(iter.dtype(i));
      }
      // If origin tensor types are satisfied with cnnl op
      // mixed types, we don't need to check with fix output
      // types.
      if (getOutputTypeWithMixedInputTypes(item_types)) {
        this->compute_dtype_ = common_dtype;
        this->fast_path = true;
        return;
      }
      // 2) Using fix output dtype to check with mixed types list.
      // modify output type to common type. And output tensor
      // target dtype is modified to common dtype.
      if (this->fix_output_dtype_ != at::ScalarType::Undefined) {
        for (int i = 0; i < n_outputs; ++i) {
          item_types[n_inputs + i] = this->fix_output_dtype_;
        }
        if (getOutputTypeWithMixedInputTypes(item_types)) {
          // Now op only support one output.
          iter.operand(0).target_dtype = this->fix_output_dtype_;
          this->compute_dtype_ = common_dtype;
          this->fast_path = false;
          return;
        }
      }
      // 3) Using common type to check with mixed types list.
      // modify output type to common type. And output tensor
      // target dtype is modified to common dtype.
      for (int i = 0; i < n_outputs; ++i) {
        item_types[n_inputs + i] = common_dtype;
      }
      if (getOutputTypeWithMixedInputTypes(item_types)) {
        // Now op only support one output.
        iter.operand(0).target_dtype = common_dtype;
        this->compute_dtype_ = common_dtype;
        this->fast_path = false;
        return;
      }
    }
  }
  // common compute type promote.
  this->compute_dtype_ = get_catch_promote_types(
      common_dtype,
      params.support_types_,
      op_name,
      /* convert_dtype */ params.allow_implicit_type_convert_);
  // Set undefined fix output type to compute dtype.
  if (this->fix_output_dtype_ == at::ScalarType::Undefined) {
    this->fix_output_dtype_ = this->compute_dtype_;
  }
  // Set fast_path to true maybe a lot wried, but it's ok.
  // We always call compute_types function first in TensorIteratorBridge,
  // so do not worry about the information being forcibly rewritten.
  this->fast_path = true;
  for (int i = 0; i < iter.ntensors(); ++i) {
    auto& op = iter.operand(i);
    if (op.is_output == true) {
      if (op.current_dtype == this->fix_output_dtype_)
        continue;
      this->fast_path = false;
      op.target_dtype = this->fix_output_dtype_;
    } else {
      if (op.current_dtype == this->compute_dtype_)
        continue;
      this->fast_path = false;
      op.target_dtype = this->compute_dtype_;
    }
  }
}

// Almost pytorch binary ops support cpu scalar tensor. And When the first
// operand is cpu scalar tensor, the device guard can't switch to the correct
// device. So we need to switch to the correct device manually. In Pytorch side,
// this operation is done in gpu_kernel_with_scalars function.
// https://github.com/pytorch/pytorch/blob/release/1.6/aten/src/ATen/native/cuda/Loops.cuh#L79
// But in catch side, we can't find a common function before each internal op
// call. So we add this in TensorIteratorBridge.
void TensorIteratorBridge::switch_to_correct_device(
    at::TensorIteratorBase& iter) {
  // First operand is cpu scalar tensor and second tensor is defined.
  // Don't need to check second tensor' device type, because TensorIterator
  // already do this check in 'compute_types'.
  if (iter.ntensors() == 3 && (iter.ntensors() - iter.noutputs() == 2) &&
      is_cpu_scalar(iter.operand(1)) &&
      iter.operand(2).tensor_base().defined()) {
    device_guard_.reset_device(iter.operand(2).tensor_base().device());
  }
}

// Note [TensorBroadcastOnTensorIteratorBridge]
// Broadcast input tensor to common shape. And this broadcast is not real
// expand, just expand value 1 in begin of tensor size and call expand op to
// change tensors size and stride. For mlu pytorch training, broadcast has two
// ways.
// 1. Catch will broadcast each tensor dims to common shape dims;
// 2. CNNL will broadcast each tensor dim value to to common shape dim value.
//   example:        a (2,3,4)   b (2,2,1,4) common shape (2,2,3,4)
//   Catch handle:   a (1,2,3,4) b (2,2,1,4)
//   CNNL handle:    a (2,2,3,4) b (2,2,3,4)
void TensorIteratorBridge::input_tensor_broadcast(
    at::TensorIteratorBase& iter,
    const CnnlOpParams& params) {
  if (params.close_tensor_broadcast_ == true)
    return;
  auto noutputs = iter.noutputs();
  auto ntensors = iter.ntensors();
  for (int i = noutputs; i < ntensors; ++i) {
    auto& op = iter.operand(i);
    TORCH_CHECK(op.tensor_base().defined(), "Input tensor is not defined.");
    TORCH_CHECK(
        !op.original_tensor_base().defined(),
        "Input original_tensor is defined.");
    // Now using cnnl op broadcast function, just expand value 1 in begin of
    // tensor size. Or maybe expand tensor to common shape directly, in this
    // situation device memory usage will be arising, but performance will be
    // arising too. So this requires a trade-off. Now we just expand value 1 in
    // begin of tensor size.
    const int tensor_ndim = op.tensor_base().dim();
    TORCH_CHECK(
        this->ndim_ >= tensor_ndim, "Output dim is less than input dim.");
    if (!is_cpu_scalar(op) && this->ndim_ > tensor_ndim) {
      std::vector<int64_t> shape = op.tensor_base().sizes().vec();
      shape.insert(shape.begin(), this->ndim_ - tensor_ndim, 1);
      // Reduce at::expand on mlu profile timechart for align ops stack same
      // with original pytorch. This operation has no impact on performance.
      // TensorIteratorBridge.cpp file is compiled before cnnl_expand file,
      // so cnnl_expand is not available.
      // Write a mirror code to instead cnnl_expand and cnnl_as_stride.
      std::vector<int64_t> expandedSizes;
      std::vector<int64_t> expandedStrides;
      std::tie(expandedSizes, expandedStrides) = at::inferExpandGeometry(
          op.tensor_base().sizes(), op.tensor_base().strides(), shape);
      auto* self_impl = getMluTensorImpl(op.tensor());
      // generate a broadcast tensor. More details in cnnl_as_stride.cpp.
      auto broadcast_tensor = at::detail::make_tensor<c10::TensorImpl>(
          c10::TensorImpl::VIEW,
          c10::Storage(op.tensor_base().storage()),
          op.tensor_base().key_set(),
          op.tensor_base().dtype());
      // Set size and stride.
      auto* broadcast_impl = getMluTensorImpl(broadcast_tensor);
      broadcast_impl->set_sizes_and_strides(
          expandedSizes, expandedStrides, op.tensor_base().storage_offset());
      // Move broadcast_tensor to op.tensor.
      update_operand_tensor_info(op, std::move(broadcast_tensor));
    }
  }
}

/**
 * Note [CATCH FAST_SETUP_TYPE]
 * CATCH fast setup type is only based on is_non_overlapping_and_dense input
 * tensors. And the fast setup type priority is: channels_last >
 * channels_last_3d > contiguous > is_non_overlapping_and_dense. 1) Collect
 * tensor list from input tensors without is_non_overlapping_and_dense tensor
 * and cpu scalar tensor; 2) Call compute_tensors_setup_type to get mlu setup
 * type for tensor list; 3) If is_all_non_overlapping_and_dense == true and
 * setup_type is NON_OVERLAPPING_DENSE, using setup type NON_OVERLAPPING_DENSE.
 * Otherwise fall back to channels_last_3d or channels_last or contiguous.
 *
 */
/**
 * Note [COMPUTE_MLU_SETUP_TYPE]
 * COMPUTE_MLU_SETUP_TYPE is based on op memory format strategy.
 * 1) If op memory format strategy is OpMluMemoryFormatStrategy::Contiguous or
 *    OpMluMemoryFormatStrategy::ChannelsLast , then using op memory format to
 *    get TensorIteratorBridge memory format;
 *    If op memory format strategy is
 * OpMluMemoryFormatStrategy::SuggestMemoryFormatHard, then using input first
 * tensor's suggest_memroy_format to get TensorIteratorBridge memory format; 2)
 * If op memory format strategy is OpMluMemoryFormatStrategy::Default or
 *    OpMluMemoryFormatStrategy::SuggestMemoryFormatSoft, then first using fast
 *    setup type to get TensorIteratorBridge memory format, if fast setup type
 *    is not supported, then fallback c10::MemoryFormat::Contiguous,
 *    c10::MemoryFormat::ChannelsLast or c10::MemoryFormat::ChannelsLast3d based
 *    on op memory format strategy and common dim value.
 *
 */
void TensorIteratorBridge::compute_mlu_setup_type(
    const at::TensorIteratorBase& iter,
    const CnnlOpParams& params) {
  auto opSuggestMemoryFormat = params.memory_format_strategy_;
  auto noutputs = iter.noutputs();
  {
    // Using Op Contiguous or ChannelsLast strategy.
    if (opSuggestMemoryFormat == OpMluMemoryFormatStrategy::ChannelsLast) {
      if (this->ndim_ == 4) {
        this->memory_format_ = c10::MemoryFormat::ChannelsLast;
      } else if (this->ndim_ == 5) {
        this->memory_format_ = c10::MemoryFormat::ChannelsLast3d;
      } else {
        // Fallback to Contiguous strategy.
        opSuggestMemoryFormat = OpMluMemoryFormatStrategy::Contiguous;
      }
    } else if (opSuggestMemoryFormat == OpMluMemoryFormatStrategy::Contiguous) {
      this->memory_format_ = c10::MemoryFormat::Contiguous;
    } else if (
        opSuggestMemoryFormat ==
        OpMluMemoryFormatStrategy::SuggestMemoryFormatHard) {
      // first input tensor memory format.
      this->memory_format_ =
          iter.operand(iter.noutputs()).tensor_base().suggest_memory_format();
      // fallback to c10::MemoryFormat::ChannelsLast3d when dim value is 5 and
      // first tensor suggest memory format is c10::MemoryFormat::ChannelsLast.
      if (this->memory_format_ == c10::MemoryFormat::ChannelsLast &&
          this->ndim_ == 5) {
        this->memory_format_ = c10::MemoryFormat::ChannelsLast3d;
      }
    }

    // iter.is_contiguous() is mean all tensors are non_overlapping_and_dense
    if (iter.is_contiguous() && this->fast_path == true) {
      // Op memory format is decided by TensorIteratorBridge.
      if (opSuggestMemoryFormat == OpMluMemoryFormatStrategy::Default ||
          opSuggestMemoryFormat ==
              OpMluMemoryFormatStrategy::SuggestMemoryFormatSoft) {
        return;
      } else {
        // Op memory format is assigned in cnnlOpParams. Check memory format is
        // same with input tensor.
        auto& input_op = iter.operand(noutputs);
        // this->memory_format_ is initialized when opSuggestMemoryFormat is
        // Contiguous or ChannelsLast.
        if (input_op.tensor_base().is_contiguous(this->memory_format_)) {
          return;
        }
      }
    }
    // Set fast_path to false.
    this->fast_path = false;
    if (opSuggestMemoryFormat == OpMluMemoryFormatStrategy::ChannelsLast ||
        opSuggestMemoryFormat == OpMluMemoryFormatStrategy::Contiguous ||
        opSuggestMemoryFormat ==
            OpMluMemoryFormatStrategy::SuggestMemoryFormatHard) {
      this->strides_ =
          std::move(get_contiguous_strides(this->shape_, this->memory_format_));
      return;
    }
  }

  // Using op Default or SuggestMemoryFormatSoft strategy
  auto ntensors = iter.ntensors();
  bool is_all_non_overlapping_and_dense = true;
  std::vector<at::TensorBase> tensor_vec;
  tensor_vec.reserve((ntensors - noutputs));
  for (int i = noutputs; i < ntensors; ++i) {
    auto& op = iter.operand(i);
    TORCH_CHECK(op.tensor_base().defined(), "Input tensor is not defined.");
    // Skip scalar tensor in mlu setup type decision.
    if (is_cpu_scalar(op))
      continue;
    // Overlapping or not dense tensor don't need to compute tensors
    // mlu setup type.
    if (!op.tensor_base().is_non_overlapping_and_dense()) {
      is_all_non_overlapping_and_dense = false;
      continue;
    }
    tensor_vec.push_back(op.tensor_base());
  }
  // TODO(shangang): 1) Align c10::MemoryFormat::ChannelsLast fallback to
  //                    c10::MemoryFormat::ChannelsLast;
  //                 2) Align using op memory format strategy instead of
  //                    MluFastSetupType.
  auto fast_setup_type_ = compute_tensors_setup_type(tensor_vec);
  // Setup type is decided by is_non_overlapping_and_dense tensors, so need
  // to check with common shape dim.
  // 1) Dim size is greater than 5, but setup type is CHANNELS_LAST or
  // CHANNELS_LAST_3D,
  //    fallback to contiguous.
  // 2) Dim size is 5, but only 4-dim tensor is CHANNELS_LAST contiguous, so
  // fallback
  //    to CHANNELS_LAST_3D.
  if (fast_setup_type_ == MluFastSetupType::CHANNELS_LAST ||
      fast_setup_type_ == MluFastSetupType::CHANNELS_LAST_3D) {
    if (this->ndim_ == 5) {
      this->memory_format_ = c10::MemoryFormat::ChannelsLast3d;
    } else if (this->ndim_ == 4) {
      this->memory_format_ = c10::MemoryFormat::ChannelsLast;
    } else {
      // Fallback to contiguous.
      fast_setup_type_ = MluFastSetupType::CONTIGUOUS;
    }
  }
  if (is_all_non_overlapping_and_dense == false &&
      fast_setup_type_ == MluFastSetupType::NON_OVERLAPPING_DENSE) {
    // (TODO) shangang:
    // 1) Maybe more effective way is to move tensor MluFastSetupType to
    // NON_OVERLAPPING_DENSE, ant call cnnl_contiguous to get other tensors to
    // NON_OVERLAPPING_DENSE. Now cnnl_contiguous is not support this. So set
    // this value to None, and get memory format based on op memory format
    // strategy. 2) Maybe using memory format of first tensor in tensor_vec is
    // better.
    fast_setup_type_ = MluFastSetupType::NONE;
  }
  // Get memory format based on op Default or SuggestMemoryFormatSoft strategy.
  if (fast_setup_type_ == MluFastSetupType::NONE) {
    if (opSuggestMemoryFormat ==
        OpMluMemoryFormatStrategy::SuggestMemoryFormatSoft) {
      // first input tensor memory format.
      this->memory_format_ =
          iter.operand(noutputs).tensor_base().suggest_memory_format();
      // fallback to c10::MemoryFormat::ChannelsLast3d when dim value is 5 and
      // first tensor suggest memory format is c10::MemoryFormat::ChannelsLast.
      if (this->memory_format_ == c10::MemoryFormat::ChannelsLast &&
          this->ndim_ == 5) {
        this->memory_format_ = c10::MemoryFormat::ChannelsLast3d;
      }
    } else {
      // Default strategy, fallback to CL contiguous when dim value is 4 or 5.
      this->memory_format_ = c10::MemoryFormat::Contiguous;
      if (this->ndim_ == 4 || this->ndim_ == 5) {
        this->memory_format_ =
            torch_mlu::get_channels_last_memory_format(this->ndim_);
      }
    }
  }
  // memory_format_ default value is contiguous, just add this for more
  // readable.
  if (fast_setup_type_ == MluFastSetupType::CONTIGUOUS) {
    this->memory_format_ = c10::MemoryFormat::Contiguous;
  }
  // 1) Using first tensor stride size when setup_type is NON_OVERLAPPING_DENSE;
  // 2) Calculation strides based on memory format.
  if (fast_setup_type_ == MluFastSetupType::NON_OVERLAPPING_DENSE) {
    // TensorIteratorBridge memory format is using Preserve to mark
    // non overlapping and dense.
    this->memory_format_ = c10::MemoryFormat::Preserve;
    this->strides_ = std::move(tensor_vec[0].strides().vec());
  } else {
    this->strides_ =
        std::move(get_contiguous_strides(this->shape_, this->memory_format_));
  }
}

// Based on TensorIteratorBridge memory format and stride, modify input tensors
// size and stride for cnnl kernel compute. If common memory format is Preserve,
// this mean all input tensor is NON_OVERLAPPING_DENSE, just need to check
// tensor types. Otherwise call cnnl_contiguous to get a new tensor, and then
// check tensor types.
void TensorIteratorBridge::resize_or_cast_input_tensor(
    at::TensorIteratorBase& iter,
    const CnnlOpParams& params) {
  const bool is_non_overlapping_dense =
      this->memory_format_ == c10::MemoryFormat::Preserve;
  auto noutputs = iter.noutputs();
  auto ntensors = iter.ntensors();
  for (int i = noutputs; i < ntensors; ++i) {
    auto& op = iter.operand(i);
    // Skip cpu scalar tensor in contiguous.
    if (is_cpu_scalar(op))
      continue;
    const bool is_same_dtype = is_same_current_and_target_dtype(op);
    const bool is_same_memory_format = is_non_overlapping_dense ||
        op.tensor_base().is_contiguous(this->memory_format_);
    if (is_same_dtype && is_same_memory_format)
      continue;
    if (is_same_dtype && !is_same_memory_format) {
      at::Tensor internal_tensor =
          torch_mlu::cnnl_contiguous(op.tensor(), this->memory_format_);
      update_operand_tensor_info(op, std::move(internal_tensor));
    } else if (!is_same_dtype && is_same_memory_format) {
      at::Tensor internal_tensor = op.tensor().to(op.target_dtype);
      update_operand_tensor_info(op, std::move(internal_tensor));
    } else {
      // If tensor is not contiguous and dtype is not same with target dtype,
      // get contiguous tensor first, and then cast to target dtype.
      auto temp_tensor =
          torch_mlu::cnnl_contiguous(op.tensor(), this->memory_format_);
      update_operand_tensor_info(op, temp_tensor.to(op.target_dtype));
    }
  }
}

// Based on output operand info, create a new tensor or resize tensor.
// prerequisite:
//   1. output tensor is defined;
//   2. current dtype is pytorch decided; and target dtype is needed by
//      cnnl op kernel.
// Output tensor is define, op.original_tensor need be set, and there
// will be have two different situation:
//   2.1 will_resize == True, this mean output tensor can be modified,
//       so call set_output to modify tensor info stored in op struct;
//   2.2 will_resize == False, this mean output tensor can't be modified.
//       So check the output tensor whether satisfied common setup type,
//       if satisfied, just using output tensor, otherwise need to create
//       a new tensor with common setup type.
void TensorIteratorBridge::malloc_or_resize_output_tensor(
    at::TensorIteratorBase& iter) {
  auto noutputs = iter.noutputs();
  for (int i = 0; i < noutputs; ++i) {
    auto& op = iter.operand(i);
    const auto& tensor_size = op.tensor_base().sizes();
    // current_dtype and target_dtype are setted when operand initialized.
    if (op.will_resize == true) {
      // Call set_output interface to resize output tensor.
      iter.set_output_raw_strided(
          i,
          tensor_size,
          this->strides_,
          original_options(op, op.current_dtype),
          iter.get_dim_names());
      // Check current and compute dtype, if not same, create a new tensor for
      // kernel compute.
      if (!is_same_current_and_target_dtype(op)) {
        at::Tensor internal_tensor = torch_mlu::mlu::empty_strided(
            tensor_size, this->strides_, original_options(op, op.target_dtype));
        update_operand_tensor_info(op, std::move(internal_tensor));
      }
    } else {
      // If output tensor is inplace of first input tensor, so no need to create
      // a new one. So just point to the first input tensor. Pytorch just
      // support one output tensor inplace to first input tensor. If support
      // multi-output tensors inplace input tensors, need to modify here.
      if (op.is_read_write == true) {
        // Find inplace input tensor.
        // 1) If any input tensor is same with this output tensor, this mean
        // inplace tensor
        //    is not changed, just using output tensor.
        const auto& ntensors = iter.ntensors();
        bool is_inplace_tensor_changed = true;
        for (int j = noutputs; j < ntensors; ++j) {
          if (iter.operand(j).tensor().is_same(op.tensor())) {
            is_inplace_tensor_changed = false;
            break;
          }
        }
        if (is_inplace_tensor_changed == false)
          continue;
        // 2) If inplace input tensor is changed, we need to find this inplace
        // input tensor,
        //    and reuse this input operand info for output.
        int inplace_index = 0;
        for (int j = noutputs; j < ntensors; ++j) {
          const auto& input_op = iter.operand(j);
          if (input_op.original_tensor_base().defined() &&
              input_op.original_tensor().is_same(op.tensor())) {
            inplace_index = j;
            break;
          }
        }
        TORCH_CHECK(
            inplace_index != 0,
            "Can't find a inplace tensor when output operand is_read_write flag is true,");
        auto& inplace_op = iter.operand(inplace_index);
        // op.original_tensor is undefined, and already checked in to_build.
        // Using input inplace op info to reduce cast op or IO kernel op of
        // output tensor.
        op.exchange_tensor(
            c10::MaybeOwned<at::TensorBase>::borrowed(inplace_op.tensor()));
        op.current_dtype = inplace_op.current_dtype;
        op.target_dtype = inplace_op.target_dtype;
      } else {
        // TODO(shangang): output tensor will resize flag is False when size is
        // same with common shape in pytorch. This will cause output original
        // tensor will be a strided tensor, and using copy with stride to copy
        // data from output tensor to output original tensor. If output tensor
        // size and stride are same with common size and stride, and current
        // dtype is same with target dtype, this tensor can be used in kernel
        // launch. Otherwise need to malloc a new tensor for output.
        // Normally pytorch TensorIterator has already checked size.
        if (!is_same_current_and_target_dtype(op) ||
            is_same_stride(op) == false) {
          at::Tensor internal_tensor = torch_mlu::mlu::empty_strided(
              tensor_size,
              this->strides_,
              original_options(op, op.target_dtype));
          update_operand_tensor_info(op, std::move(internal_tensor));
        }
      }
    }
  }
}

// Op support stride overlap, so don't need to check tensor stride, just check
// the tensor dtype. Add stride function here is to avoid unnecessary code
// execute.
void TensorIteratorBridge::cast_input_output_tensors_with_stride(
    at::TensorIteratorBase& iter) {
  auto noutputs = iter.noutputs();
  auto ntensors = iter.ntensors();
  // Cast input tensors.
  for (int i = noutputs; i < ntensors; ++i) {
    auto& op = iter.operand(i);
    const bool is_same_dtype = is_same_current_and_target_dtype(op);
    // Skip cpu scalar tensor in contiguous.
    if (is_cpu_scalar(op) || is_same_dtype)
      continue;
    // CNNL Cast op is not support stride, so we will get a contiguous tensor.
    update_operand_tensor_info(op, op.tensor().to(op.target_dtype));
  }
  // Cast output tensors.
  for (int i = 0; i < noutputs; ++i) {
    auto& op = iter.operand(i);
    if (is_cpu_scalar(op))
      continue;
    // If output tensor is inplace of first input tensor, so no need to create a
    // new one. So just point to the first input tensor. Pytorch just support
    // one output tensor inplace to first input tensor. If support multi-output
    // tensors inplace input tensors, need to modify here.
    if (op.is_read_write == true) {
      // Find inplace input tensor.
      // 1) If any input tensor is same with this output tensor, this mean
      // inplace tensor
      //    is not changed, just using output tensor.
      const auto& ntensors = iter.ntensors();
      bool is_inplaced_tensor_changed = true;
      for (int j = noutputs; j < ntensors; ++j) {
        if (iter.operand(j).tensor().is_same(op.tensor())) {
          is_inplaced_tensor_changed = false;
          break;
        }
      }
      if (is_inplaced_tensor_changed == false)
        continue;
      // 2) If inplaced input tensor is changed, we need to find this inplaced
      // input tensor,
      //    and reuse this input operand info for output.
      int inplace_index = 0;
      for (int j = noutputs; j < ntensors; ++j) {
        const auto& input_op = iter.operand(j);
        if (input_op.original_tensor_base().defined() &&
            input_op.original_tensor().is_same(op.tensor())) {
          inplace_index = j;
          break;
        }
      }
      TORCH_CHECK(
          inplace_index != 0,
          "Can't find a inplace tensor when output operand is_read_write flag is true,");
      auto& inplace_op = iter.operand(inplace_index);
      // op.original_tensor is undefined, and already checked in to_build.
      // Using input inplace op info to reduce cast op or IO kernel op of output
      // tensor.
      op.exchange_tensor(
          c10::MaybeOwned<at::TensorBase>::borrowed(inplace_op.tensor()));
      op.current_dtype = inplace_op.current_dtype;
      op.target_dtype = inplace_op.target_dtype;
    } else if (!is_same_current_and_target_dtype(op)) {
      // CNNL Cast op is not support stride, so we will get a contiguous tensor.
      update_operand_tensor_info(op, op.tensor().to(op.target_dtype));
    }
  }
}

void TensorIteratorBridge::to_build(
    at::TensorIteratorBase& iter,
    const std::string& op_name) {
  // All Tensor in TensorIterator need to be defined after mlu add a patch in
  // TensorIteratorBase::build function. That patch is to set a operand
  // will_resize flag is true when malloc a new output tensor (which is not
  // defined in user side.) with common dtype and common shape. After this
  // patch, all output tensor will be defined.
  for (int i = 0; i < iter.noutputs(); ++i) {
    auto& op = iter.operand(i);
    TORCH_CHECK(
        !op.original_tensor_base().defined(),
        "Output original_tensor is defined.");
    TORCH_CHECK(op.tensor_base().defined(), "Output tensor is not defined.");
  }
  // Update ndim of TensorIteratorBridge.
  // common shape and ndim has been coalesced in pytorch side. Output tensor
  // always created by common shape, so using first output tensor size to
  // broadcast input tensor.
  this->ndim_ = iter.operand(0).tensor_base().dim();
  this->shape_ = iter.operand(0).tensor_base().sizes();
  // Get fix output dtype and align with gpu side for better performance.
  // Like logic op using fixed bool as output dtype.
  // TensorIteratorBase is build by build_borrowing_unary_force_boolean_op
  // or build_comparison_op or build_borrowing_comparison_op.
  // Also this configuration is a little werid, cause just a few ops
  // support mixed inputs, almost cnnl op kernel need same dtype in support
  // types.
  this->fix_output_dtype_ = iter.get_static_dtype().has_value()
      ? iter.get_static_dtype().value()
      : at::ScalarType::Undefined;
  // Get cnnl op params from op name.
  const auto& params = getCnnlOpParams(op_name);
  // nullary op can be handled simply. Avoid error call to common_dtype().
  if (nullary_input(iter, params, op_name))
    return;
  // switch to mlu correct device
  switch_to_correct_device(iter);
  // compute the result dtype that be support in mlu.
  compute_types(iter, params, op_name);
  // Support strided memory.
  if (params.allow_strided_memory_ == true) {
    // broadcast if necessary
    input_tensor_broadcast(iter, params);
    // Only cast tensor to target dtype and keep tensor stride.
    // This need cnnl cast kernel support stride.
    cast_input_output_tensors_with_stride(iter);
    return;
  }
  // compute mlu setup type based on input tensors
  compute_mlu_setup_type(iter, params);
  // broadcast if necessary
  input_tensor_broadcast(iter, params);
  if (this->fast_path == true)
    return;
  // cast or contiguous mlu input tensors.
  resize_or_cast_input_tensor(iter, params);
  // malloc or resize output tensor
  malloc_or_resize_output_tensor(iter);
}

} // namespace torch_mlu
