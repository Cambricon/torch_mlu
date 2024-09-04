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

} // namespace torch_mlu
