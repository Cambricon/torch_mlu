
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
#include "ATen/native/BinaryOps.h"
#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {
namespace ops {
using namespace at::native;
/***************************************common****************************************/
void common_mlu_kernel(
    at::Tensor& output,
    const at::Tensor& input_,
    const at::Tensor& other_,
    cnnlLogicOp_t logic_type,
    at::ScalarType& compute_dtype) {
  auto input = cast_long_to_int_if_needed(input_);
  auto other = cast_long_to_int_if_needed(other_);
  auto out = create_int_tensor_if_needed(output);
  cnnl_logic_internal(out, input, other, logic_type, compute_dtype);
  cast_int_to_long_if_needed(out, output);
}

/***************************************eq/ne****************************************/
void eq_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto compute_type = iter.common_dtype();
  cnnl_logic_internal(
      output, iter.input(0), iter.input(1), CNNL_LOGIC_OP_EQ, compute_type);
  iter.cast_outputs();
}

void ne_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto compute_type = iter.common_dtype();
  cnnl_logic_internal(
      output, iter.input(0), iter.input(1), CNNL_LOGIC_OP_NE, compute_type);
  iter.cast_outputs();
}

bool cnnl_equal(const at::Tensor& self, const at::Tensor& other) {
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), other.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(
      self.device() == other.device(),
      "Cannot compare two tensors on "
      "different devices. Got: ",
      self.device(),
      " and ",
      other.device());
  if (self.sizes() != other.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }
  auto output_tensor = at::eq(self, other);
  return output_tensor.all().item().to<bool>();
}
/***************************************isnan****************************************/
at::Tensor cnnl_isnan(const at::Tensor& self) {
  return at::native::isnan(self);
}
/***************************************le/lt/ge/gt****************************************/
void le_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto compute_type = iter.common_dtype();
  cnnl_logic_internal(
      output, iter.input(0), iter.input(1), CNNL_LOGIC_OP_LE, compute_type);
  iter.cast_outputs();
}

void lt_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto compute_type = iter.common_dtype();
  cnnl_logic_internal(
      output, iter.input(0), iter.input(1), CNNL_LOGIC_OP_LT, compute_type);
  iter.cast_outputs();
}

void ge_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto compute_type = iter.common_dtype();
  cnnl_logic_internal(
      output, iter.input(0), iter.input(1), CNNL_LOGIC_OP_GE, compute_type);
  iter.cast_outputs();
}

void gt_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto compute_type = iter.common_dtype();
  cnnl_logic_internal(
      output, iter.input(0), iter.input(1), CNNL_LOGIC_OP_GT, compute_type);
  iter.cast_outputs();
}
/***************************************logical_not****************************************/
void logical_not_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0)
    return;
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "logical_not");
  auto output = iter_bridge.output(iter, 0);
  auto compute_type = iter.common_dtype();
  compute_type = compute_type == at::kLong ? at::kInt : compute_type;
  common_mlu_kernel(
      output,
      iter_bridge.input(iter, 0),
      iter_bridge.input(iter, 0),
      CNNL_LOGIC_OP_NOT,
      compute_type);
  iter.cast_outputs();
}

at::Tensor& cnnl_logical_not_out(const at::Tensor& self, at::Tensor& out) {
  at::TensorIteratorConfig config;
  config.check_all_same_dtype(false);
  config.add_output(out);
  config.add_input(self);
  // When 'out' isn't defined (e.g. torch.logical_not(a)), we want the output to
  // be bool. When 'out' is defined (e.g. 'torch.logical_not(a, out=b)'), we
  // want that all kernels using this TensorIterator will need to special-case
  // when the output tensor has bool dtype, and provide a lambda of type
  // (scalar_t -> bool).
  if (!out.defined() || (out.defined() && out.scalar_type() == at::kBool)) {
    config.declare_static_dtype(at::kBool);
  }
  at::TensorIterator iter = config.build();
  logical_not_mlu_kernel(iter);
  return out;
}
/***************************************logical_xor****************************************/
at::Tensor& cnnl_logical_xor_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  return at::native::logical_xor_out(self, other, out);
}

void logical_xor_mlu_kernel(at::TensorIterator& iter) {
  if (iter.numel() == 0)
    return;
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "logical_xor");
  auto output = iter_bridge.output(iter, 0);
  auto compute_type = iter.common_dtype();
  compute_type = compute_type == at::kLong ? at::kInt : compute_type;
  common_mlu_kernel(
      output,
      iter_bridge.input(iter, 0),
      iter_bridge.input(iter, 1),
      CNNL_LOGIC_OP_XOR,
      compute_type);
  iter.cast_outputs();
}
/***************************************logical_and****************************************/
at::Tensor& cnnl_logical_and_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  return at::native::logical_and_out(self, other, out);
}

void logical_and_mlu_kernel(at::TensorIterator& iter) {
  if (iter.numel() == 0)
    return;
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "logical_and");
  auto output = iter_bridge.output(iter, 0);
  auto compute_type = iter.common_dtype();
  compute_type = compute_type == at::kLong ? at::kInt : compute_type;
  common_mlu_kernel(
      output,
      iter_bridge.input(iter, 0),
      iter_bridge.input(iter, 1),
      CNNL_LOGIC_OP_AND,
      compute_type);
  iter.cast_outputs();
}
/***************************************logical_or****************************************/
at::Tensor& cnnl_logical_or_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
  return at::native::logical_or_out(self, other, out);
}

void logical_or_mlu_kernel(at::TensorIterator& iter) {
  if (iter.numel() == 0)
    return;
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "logical_or");
  auto output = iter_bridge.output(iter, 0);
  auto compute_type = iter.common_dtype();
  compute_type = compute_type == at::kLong ? at::kInt : compute_type;
  common_mlu_kernel(
      output,
      iter_bridge.input(iter, 0),
      iter_bridge.input(iter, 1),
      CNNL_LOGIC_OP_OR,
      compute_type);
  iter.cast_outputs();
}
REGISTER_PRIVATEUSE1_DISPATCH(eq_stub, &eq_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(ne_stub, &ne_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(le_stub, &le_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(lt_stub, &lt_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(ge_stub, &ge_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(gt_stub, &gt_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(logical_not_stub, &logical_not_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(logical_xor_stub, &logical_xor_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(logical_or_stub, &logical_or_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(logical_and_stub, &logical_and_mlu_kernel);
} // namespace ops
} // namespace torch_mlu
