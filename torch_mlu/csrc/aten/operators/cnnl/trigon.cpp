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

#include "ATen/native/UnaryOps.h"
#include "ATen/native/BinaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

void trigon_kernel_mlu(
    at::TensorIteratorBase& iter,
    cnnlTrigonFunctionMode_t mode) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_trigon_internal(output, self, mode);
  iter.cast_outputs();
}

void sin_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_SIN);
}

void cos_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_COS);
}

void tan_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_TAN);
}

void asinh_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_ASINH);
}

void acosh_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_ACOSH);
}

void atanh_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_ATANH);
}

void asin_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_ASIN);
}

void acos_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_ACOS);
}

void atan_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_ATAN);
}

void sinh_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_SINH);
}

void cosh_kernel_mlu(at::TensorIteratorBase& iter) {
  trigon_kernel_mlu(iter, CNNL_TRIGON_COSH);
}

void atan2_kernel_mlu(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto self = iter.input(0);
  auto other = iter.input(1);
  // Can't call common_dtype() here, using output tensor dtype instead.
  // Cause output dtype is same with common dtype.
  auto self_tensor = scalar_to_tensor_with_dtype(self, output.scalar_type());
  auto other_tensor = scalar_to_tensor_with_dtype(other, output.scalar_type());
  cnnl_atan2_internal(output, self_tensor, other_tensor);
  iter.cast_outputs();
}

using namespace at::native;
REGISTER_PRIVATEUSE1_DISPATCH(sin_stub, &sin_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(cos_stub, &cos_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(tan_stub, &tan_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(asinh_stub, &asinh_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(acosh_stub, &acosh_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(atanh_stub, &atanh_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(asin_stub, &asin_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(acos_stub, &acos_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(atan_stub, &atan_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(sinh_stub, &sinh_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(cosh_stub, &cosh_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(atan2_stub, &atan2_kernel_mlu);

} // namespace ops
} // namespace torch_mlu
