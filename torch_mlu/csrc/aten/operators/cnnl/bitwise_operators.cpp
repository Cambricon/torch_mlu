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

#include "ATen/ScalarOps.h"
#include "ATen/native/BinaryOps.h"
#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/utils/binaryops_util.h"
#include "aten/utils/dispatch.h"
#include "aten/TensorIteratorBridge.h"

namespace torch_mlu {
namespace ops {

using at::native::bitwise_and_stub;
using at::native::bitwise_and_stub_DECLARE_DISPATCH_type;
using at::native::bitwise_not_stub;
using at::native::bitwise_not_stub_DECLARE_DISPATCH_type;
using at::native::bitwise_or_stub;
using at::native::bitwise_or_stub_DECLARE_DISPATCH_type;
using at::native::bitwise_xor_stub;
using at::native::bitwise_xor_stub_DECLARE_DISPATCH_type;
using at::native::lshift_stub;
using at::native::lshift_stub_DECLARE_DISPATCH_type;
using at::native::rshift_stub;
using at::native::rshift_stub_DECLARE_DISPATCH_type;

void bitwise_operators_impl(
    at::TensorIteratorBase& iter,
    cnnlBitComputeOp_t op) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  auto other = iter.input(1);

  // self and other must have the same dtype as output
  if (other.scalar_type() != output.scalar_type()) {
    other = other.to(output.scalar_type());
  }
  if (self.scalar_type() != output.scalar_type()) {
    self = self.to(output.scalar_type());
  }
  if (op == CNNL_CYCLE_BAND_OP || op == CNNL_CYCLE_BOR_OP ||
      op == CNNL_CYCLE_BXOR_OP) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(
        at::kBool, iter.dtype(), "mlu bitwise logic op:", [&]() {
          cnnl_bitwise_op_out_internal(output, self, other, op);
        });
  } else if (op == CNNL_BLEFT_SHIFT_OP_V2 || op == CNNL_BRIGHT_SHIFT_OP_V2) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "mlu bitwise shift op:", [&]() {
      cnnl_bitwise_op_out_internal(output, self, other, op);
    });
  }
}

void bitwise_and_mlu_kernel(at::TensorIteratorBase& iter) {
  bitwise_operators_impl(iter, CNNL_CYCLE_BAND_OP);
}

void bitwise_or_mlu_kernel(at::TensorIteratorBase& iter) {
  bitwise_operators_impl(iter, CNNL_CYCLE_BOR_OP);
}

void bitwise_xor_mlu_kernel(at::TensorIteratorBase& iter) {
  bitwise_operators_impl(iter, CNNL_CYCLE_BXOR_OP);
}

void lshift_mlu_kernel(at::TensorIteratorBase& iter) {
  bitwise_operators_impl(iter, CNNL_BLEFT_SHIFT_OP_V2);
}

void rshift_mlu_kernel(at::TensorIteratorBase& iter) {
  bitwise_operators_impl(iter, CNNL_BRIGHT_SHIFT_OP_V2);
}

void bitwise_not_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  at::Tensor input2;
  cnnl_bitwise_op_out_internal(output, self, input2, CNNL_BNOT_OP);
}

at::Tensor cnnl___lshift__(const at::Tensor& self, const at::Scalar& other) {
  Tensor result;
  TensorIteratorBridge iter_bridge;
  auto wrapper = at::native::wrapped_scalar_tensor(other);
  auto iter = at::TensorIterator::binary_op(result, self, wrapper);
  iter_bridge.to_build(iter, "lshift");
  lshift_mlu_kernel(iter);
  iter_bridge.cast_outputs(iter);
  return iter.output();
}

at::Tensor& cnnl___ilshift__(at::Tensor& self, const at::Scalar& other) {
  TensorIteratorBridge iter_bridge;
  auto wrapper = at::native::wrapped_scalar_tensor(other);
  auto iter = at::TensorIterator::binary_op(self, self, wrapper);
  iter_bridge.to_build(iter, "lshift");
  lshift_mlu_kernel(iter);
  iter_bridge.cast_outputs(iter);
  return self;
}

at::Tensor cnnl___lshift__(const at::Tensor& self, const at::Tensor& other) {
  Tensor result;
  TensorIteratorBridge iter_bridge;
  auto iter = at::TensorIterator::binary_op(result, self, other);
  iter_bridge.to_build(iter, "lshift");
  lshift_mlu_kernel(iter);
  iter_bridge.cast_outputs(iter);
  return iter.output();
}

at::Tensor& cnnl___ilshift__(at::Tensor& self, const at::Tensor& other) {
  TensorIteratorBridge iter_bridge;
  auto iter = at::TensorIterator::binary_op(self, self, other);
  iter_bridge.to_build(iter, "lshift");
  lshift_mlu_kernel(iter);
  iter_bridge.cast_outputs(iter);
  return self;
}

at::Tensor cnnl___rshift__(const at::Tensor& self, const at::Scalar& other) {
  Tensor result;
  TensorIteratorBridge iter_bridge;
  auto wrapper = at::native::wrapped_scalar_tensor(other);
  auto iter = at::TensorIterator::binary_op(result, self, wrapper);
  iter_bridge.to_build(iter, "rshift");
  rshift_mlu_kernel(iter);
  iter_bridge.cast_outputs(iter);
  return iter.output();
}

at::Tensor& cnnl___irshift__(at::Tensor& self, const at::Scalar& other) {
  TensorIteratorBridge iter_bridge;
  auto wrapper = at::native::wrapped_scalar_tensor(other);
  auto iter = at::TensorIterator::binary_op(self, self, wrapper);
  iter_bridge.to_build(iter, "rshift");
  rshift_mlu_kernel(iter);
  iter_bridge.cast_outputs(iter);
  return self;
}

at::Tensor cnnl___rshift__(const at::Tensor& self, const at::Tensor& other) {
  Tensor result;
  TensorIteratorBridge iter_bridge;
  auto iter = at::TensorIterator::binary_op(result, self, other);
  iter_bridge.to_build(iter, "rshift");
  rshift_mlu_kernel(iter);
  iter_bridge.cast_outputs(iter);
  return iter.output();
}

at::Tensor& cnnl___irshift__(at::Tensor& self, const at::Tensor& other) {
  TensorIteratorBridge iter_bridge;
  auto iter = at::TensorIterator::binary_op(self, self, other);
  iter_bridge.to_build(iter, "rshift");
  rshift_mlu_kernel(iter);
  iter_bridge.cast_outputs(iter);
  return self;
}

REGISTER_PRIVATEUSE1_DISPATCH(bitwise_not_stub, &bitwise_not_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(bitwise_and_stub, &bitwise_and_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(bitwise_or_stub, &bitwise_or_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(bitwise_xor_stub, &bitwise_xor_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(lshift_stub, &lshift_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(rshift_stub, &rshift_mlu_kernel);
} // namespace ops
} // namespace torch_mlu
