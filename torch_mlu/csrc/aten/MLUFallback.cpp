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

#include "aten/MLUFallback.h"
#include <ATen/native/SparseTensorUtils.h>

namespace torch_mlu {

// Register fallthrough for AutogradPrivateUse1, because
// Pytorch dose not register it in
// pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp
TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

static std::set<std::string> fallback_ops;

// ops not supported by MLU fall back to run on the CPU
void mlu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  if (fallback_ops.find(op.schema().operator_name().name) ==
      fallback_ops.end()) {
    fallback_ops.insert(op.schema().operator_name().name);
    TORCH_WARN(
        "The operator '",
        op.schema().operator_name(),
        "' is not currently supported ",
        "on the MLU backend and will fall back to run on the CPU.",
        " This may have performance implications.");
  }
  at::native::cpu_fallback(op, stack);
}

// ops fail on the MLU fall back to run on the CPU
void mlu_fail_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  if (fallback_ops.find(op.schema().operator_name().name) ==
      fallback_ops.end()) {
    fallback_ops.insert(op.schema().operator_name().name);
    TORCH_WARN(
        "The operator '",
        op.schema().operator_name(),
        "' fails on the MLU backend and will fall back to run on the CPU.",
        " This may have performance implications.",
        "You can set the environment variable `TORCH_MIN_CNLOG_LEVEL=0` ",
        "to see detailed error msg, "
        "or set the environment variable `ENABLE_FALLBACK_TO_CPU=0` to disable fallback.");
  }
  at::native::cpu_fallback(op, stack);
}

void mlu_disable_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "The operator '",
      op.schema().operator_name(),
      "' is not currently implemented ",
      "for the MLU device. If you want this op to be added, ",
      "please contact the engineer of Cambricon. ",
      "As a temporary fix, you can set the environment variable `ENABLE_FALLBACK_TO_CPU=1` ",
      "to use the CPU as a fallback for this op. WARNING: this will be slower than running natively ",
      "on MLU.");
}

bool getFallbackEnabledEnvVar() {
  static const char* enable_c_str = std::getenv("ENABLE_FALLBACK_TO_CPU");
  // enable fallback by default
  if (!enable_c_str) {
    return true;
  }
  std::string enable(enable_c_str);
  if (enable == "0" || enable == "OFF" || enable == "off") {
    return false;
  }
  return true;
}

// MLU ops fail and fallback to CPU
bool getFailFallbackEnabledEnvVar() {
  static const char* enable_c_str = std::getenv("ENABLE_MLU_FAIL_FALLBACK");
  // disable by default
  if (!enable_c_str) {
    return false;
  }
  std::string enable(enable_c_str);
  if (enable == "1" || enable == "ON" || enable == "on") {
    return true;
  }
  return false;
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  static const bool enable_mlu_fallback = getFallbackEnabledEnvVar();
  if (enable_mlu_fallback) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&mlu_fallback>());
  } else {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&mlu_disable_fallback>());
  }
}

// When some out-variant ops fail on SparseMLU,
// one of the indices or values of out may have been modified,
// which will cause some checks in the fallback to fail.
// For this case we reset these outs to empty SparseTensor.
void process_out_arguments(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  auto arguments = torch::jit::last(stack, num_arguments);

  for (const auto idx : c10::irange(arguments.size())) {
    auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      at::Tensor tensor = ivalue.toTensor();
      const c10::AliasInfo* alias_info = schema_args[idx].alias_info();
      if (alias_info != nullptr && alias_info->isWrite() &&
          tensor.is_sparse() &&
          tensor._indices().sym_size(1) != tensor._values().sym_size(0)) {
        auto* tensor_impl = at::sparse::get_sparse_impl(tensor);
        auto tensor_values = tensor_impl->values();
        auto tensor_indices = tensor_impl->indices();
        tensor_indices.resize_({tensor.sparse_dim(), 0});
        std::vector<int64_t> values_size = {0};
        auto dense_size = tensor.sizes().slice(tensor.sparse_dim());
        values_size.insert(
            values_size.end(), dense_size.begin(), dense_size.end());
        tensor_values.resize_(values_size);
      }
    } else if (ivalue.isTensorList()) {
      const c10::AliasInfo* alias_info = schema_args[idx].alias_info();
      if (alias_info != nullptr && alias_info->isWrite()) {
        for (auto tensor : ivalue.toTensorList().vec()) {
          if (tensor.is_sparse() &&
              tensor._indices().sym_size(1) != tensor._values().sym_size(0)) {
            auto* tensor_impl = at::sparse::get_sparse_impl(tensor);
            auto tensor_values = tensor_impl->values();
            auto tensor_indices = tensor_impl->indices();
            tensor_indices.resize_({tensor.sparse_dim(), 0});
            std::vector<int64_t> values_size = {0};
            auto dense_size = tensor.sizes().slice(tensor.sparse_dim());
            values_size.insert(
                values_size.end(), dense_size.begin(), dense_size.end());
            tensor_values.resize_(values_size);
          }
        }
      }
    }
  }
}

// ops not supported by SparseMLU fall back to run on the SparseCPU
void mlu_sparse_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  const std::string sparse_name = op.schema().operator_name().name + "_sparse";
  if (fallback_ops.find(sparse_name) == fallback_ops.end()) {
    fallback_ops.insert(sparse_name);
    TORCH_WARN(
        "The operator '",
        op.schema().operator_name(),
        "' is not currently supported ",
        "on the SparseMLU backend and will fall back to run on the SparseCPU.",
        " This may have performance implications.");
  }
  at::native::cpu_fallback(op, stack, false, c10::DispatchKey::SparseCPU);
}

// ops fail on the SparseMLU fall back to run on the SparseCPU
void mlu_sparse_fail_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  const std::string sparse_name = op.schema().operator_name().name + "_sparse";
  if (fallback_ops.find(sparse_name) == fallback_ops.end()) {
    fallback_ops.insert(sparse_name);
    TORCH_WARN(
        "The operator '",
        op.schema().operator_name(),
        "' fails on the SparseMLU backend and will fall back to run on the SparseCPU.",
        " This may have performance implications.",
        "You can set the environment variable `TORCH_MIN_CNLOG_LEVEL=0` ",
        "to see detailed error msg, "
        "or set the environment variable `ENABLE_FALLBACK_TO_CPU=0` to disable fallback.");
  }
  process_out_arguments(op, stack);
  at::native::cpu_fallback(op, stack, false, c10::DispatchKey::SparseCPU);
}

TORCH_LIBRARY_IMPL(_, SparsePrivateUse1, m) {
  static const bool enable_mlu_fallback = getFallbackEnabledEnvVar();
  if (enable_mlu_fallback) {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&mlu_sparse_fallback>());
  } else {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&mlu_disable_fallback>());
  }
}

} // namespace torch_mlu
