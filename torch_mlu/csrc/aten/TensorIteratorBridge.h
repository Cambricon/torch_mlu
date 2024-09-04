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

#include <c10/core/DeviceGuard.h>
#include <ATen/native/TensorIterator.h>
#include "aten/utils/tensor_util.h"
#include "aten/utils/utils.h"

/**
 * Note [TensorIteratorBridge]
 * ~~~~~~~~~~~~~~~~
 * TensorIteratorBridge is a helper class to optimize mlu element-wise
 * operations. such as arithmetic, comparisons, and trigonometric functions. It
 * handles broadcasting, type conversions of operands, and also decide best mlu
 * setup type of operations.
 *
 * Broadcasting: By comparing common shape and tensor shape, we inserting value
 * 1 to the tensor shape, and expand the tensor to this tensor shape. Which
 * handle is different with pytorch TensorIterator.
 *
 * Type conversion: By computing common type and operation support types, we get
 * a promote type to cast all tensor to same type, which need be in operation
 * support types and can be casted.
 *
 * Computability: In MLU side, element-wise kernels need contiguous tensors or
 * at least non overlapping and dense tensors. Based on this, CATCH will align
 * all input and output tensors to almost same metadata, except some broadcast
 * in mlu kernel side. By using MluFastSetupType to describe tensors metadata.
 * CATCH calls cnnl_contiguous to align all input tensors, and malloc or resize
 * to align all output tensors.
 *
 * It should be noted that MLU element wise ops prefer channels_last or
 * channels_last_3d tensors to compute. So we using channels_last or
 * channels_last_3d as default memory format when common shape dim is equal to 4
 * or 5, otherwise contiguous as default.
 *
 * After using TensorIteratorBridge, It will reduce copying and data type
 * conversion, also it will increase memory reuse.
 *
 * if op registered structured, only need to increase
 * 'maybe_cast_or_copy_output' after kernel, TensorIteratorBridge is used in
 * 'RegisterMLU.cpp' file. And these params need input to
 * 'TensorIteratorBridge.to_build()', such as op support types, are defined in
 * 'cnnlOpParams.cpp' file.
 *
 * example:
 * Structured TensorIteratorBase op.
 * void sigmoid_mlu_kernel(at::TensorIteratorBase& iter) {
 *   auto output = iter.output(0);
 *   cnnl_activation_internal(output, iter.input(0), CNNL_ACTIVATION_SIGMOID);
 *   iter.cast_outputs();
 * }
 *
 * TensorIterator op.
 * example:
 * void sigmoid_mlu_kernel(at::TensorIteratorBase& iter) {
 *   TensorIteratorBridge iter_bridge;
 *   iter_bridge.to_build(iter, op_name);
 *   // Call TensorIteratorBridge output and input interface.
 *   auto output = iter.output(0);
 *   cnnl_activation_internal(output, iter.input(0), CNNL_ACTIVATION_SIGMOID);
 *   iter.cast_outputs();
 * }
 *
 * The basic policy of op.original_tensor and op.tensor is:
 * 1) original tensor is undefined, this mean output tensor is ok for
 *    cnnl kernel to compute, here nothing need to do;
 * 2) original tensor is defined, this mean output tensor need be original
 *    tensor, and op.tensor is just for internal compute, we need copy or cast
 *    op.tensor to op.original_tensor.
 *
 * The basic policy of 64bit is:
 * 64bit is not supported in cnnl op kernel, and catch has already implicit
 * convert 64bit to 32bit in copy form cpu function. So we don't need to do
 * anything in 'TensorIteratorBridge' file, just bypass 64bit tensor to CATCH
 * op.
 *
 */

namespace torch_mlu {

class CnnlOpParams;

class TensorIteratorBridge {
 public:
  TensorIteratorBridge() {}
  ~TensorIteratorBridge() {}

  void to_build(at::TensorIteratorBase& iter, const char* op_name);
  static void cast_outputs(at::TensorIteratorBase& iter);

  inline const at::ScalarType& compute_dtype() {
    return this->compute_dtype_;
  }

 protected:
  void compute_types(
      at::TensorIteratorBase& iter,
      const CnnlOpParams& params,
      const std::string& op_name);
  bool nullary_input(
      at::TensorIteratorBase& iter,
      const CnnlOpParams& params,
      const std::string& op_name);
  void update_operand_tensor_info(at::OperandInfo& op, at::Tensor&& tensor);
  void switch_to_correct_device(at::TensorIteratorBase& iter);
  void input_tensor_broadcast(
      at::TensorIteratorBase& iter,
      const CnnlOpParams& params);
  void resize_or_cast_input_tensor(
      at::TensorIteratorBase& iter,
      const CnnlOpParams& params);
  void malloc_or_resize_output_tensor(at::TensorIteratorBase& iter);
  void cast_input_output_tensors_with_stride(at::TensorIteratorBase& iter);
  void compute_mlu_setup_type(
      const at::TensorIteratorBase& iter,
      const CnnlOpParams& params);
  bool is_cpu_scalar(const at::OperandInfo& op);
  at::TensorOptions original_options(
      const at::OperandInfo& op,
      const at::ScalarType& scalar_dtype);
  bool is_same_current_and_target_dtype(const at::OperandInfo& op);
  bool is_same_stride(const at::OperandInfo& op);

 private:
  bool fast_path = false;
  // memory_format_ is used to describe TensorIteratorBase memory format.
  // If memory format is preserve, this mean TensorIteratorBase using
  // is_non_overlapping_and_dense, otherwise TensorIteratorBase memory format
  // is as same as c10::MemoryFormat.
  c10::MemoryFormat memory_format_ = c10::MemoryFormat::Contiguous;
  // Compute type for scalar to tensor.
  at::ScalarType compute_dtype_ = at::ScalarType::Undefined;
  // Common stride
  std::vector<int64_t> strides_;
  // ndim of common shape
  int ndim_ = 0;
  // Common shape, TensorIteratorBase shape is coalesced,
  // so we need to have caculate common shape from output tensor.
  at::DimVector shape_;
  c10::OptionalDeviceGuard device_guard_;
  // Some CNNL op support fixed output dtype and ignore input dtypes. Like
  // comparsion op. So when mixed input dtypes is supported, we will set output
  // dtype to this fixed dtype.
  at::ScalarType fix_output_dtype_ = at::ScalarType::Undefined;
};

} // namespace torch_mlu
