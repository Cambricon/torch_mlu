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

#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/SparseTensorUtils.h>
#include "framework/core/tensor_impl.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "framework/core/mlu_guard.h"
#include "aten/operators/cnnl/copy.h"
#include "aten/operators/cnnl/copy_utils.h"

namespace torch_mlu {
namespace ops {

void copy_kernel_mlu(at::TensorIterator& iter, bool non_blocking) {
  AT_ASSERT(iter.ntensors() == 2);
  c10::Device dst_device = iter.device(0);
  c10::Device src_device = iter.device(1);
  // Copy on MLU (or between MLUs)
  if (dst_device.is_privateuseone() && src_device.is_privateuseone()) {
    copy_device_to_device(iter, non_blocking);
    return;
  }

  auto dst = iter.tensor(0);
  auto src = iter.tensor(1);

  // H2D: src (CPU) => dst (MLU)
  // 1. Copy the data form the CPU to the MLU device.
  // 2. If data transformation is needed. it will be done by
  //    an MLU Transpose operator.
  // 3. Copy the data onto the dst tensor.
  if (dst.device().is_privateuseone() && src.is_cpu()) {
    torch_mlu::mlu::MLUGuard guard(dst.device());
    // cnrtMemcpy don't suppoert stride.
    auto memory_format = dst.suggest_memory_format();
    at::Tensor copy_src_non_overlapping_and_dense;
    if (src.is_non_overlapping_and_dense()) {
      copy_src_non_overlapping_and_dense = src;
    } else {
      copy_src_non_overlapping_and_dense = src.contiguous(memory_format);
    }
    if ((dst.strides() == copy_src_non_overlapping_and_dense.strides()) &&
        dst.dtype() == src.dtype()) {
      copy_from_cpu(dst, copy_src_non_overlapping_and_dense, non_blocking);
    } else {
      auto src_mlu = at::empty_like(
          copy_src_non_overlapping_and_dense,
          copy_src_non_overlapping_and_dense.options().device(
              at::kPrivateUse1));
      copy_from_cpu(src_mlu, copy_src_non_overlapping_and_dense, non_blocking);
      // D2D copy
      // propagate the corrent conjugate bit
      src_mlu._set_conj(src.is_conj());
      src_mlu._set_neg(src.is_neg());
      cnnl_copy_(dst, src_mlu, non_blocking);
    }
    auto stream = getCurrentMLUStream();
    if (!non_blocking)
      stream.synchronize();
  }

  // D2H: src (MLU) => dst (CPU)
  if (dst.is_cpu() && src.device().is_privateuseone()) {
    torch_mlu::mlu::MLUGuard guard(src.device());
    // cnrtMemcpy don't suppoert stride.
    auto memory_format = dst.suggest_memory_format();
    at::Tensor src_non_overlapping_and_dense;
    if (src.is_non_overlapping_and_dense()) {
      src_non_overlapping_and_dense = src;
    } else {
      src_non_overlapping_and_dense = cnnl_contiguous(src, memory_format);
    }
    if ((dst.strides() == src_non_overlapping_and_dense.strides()) &&
        dst.dtype() == src.dtype()) {
      copy_to_cpu(dst, src_non_overlapping_and_dense, non_blocking);
    } else {
      // Note: optimize performance for D2H copy
      // Comparing to at::empty,
      // at::zeros initialize real memory immediately and get better
      // performance.
      auto src_cpu = at::empty_like(
                         src_non_overlapping_and_dense,
                         src_non_overlapping_and_dense.options()
                             .device(at::kCPU)
                             .pinned_memory(non_blocking))
                         .zero_();
      copy_to_cpu(src_cpu, src_non_overlapping_and_dense, non_blocking);
      // propagate the corrent conjugate bit.
      src_cpu._set_conj(src.is_conj());
      src_cpu._set_neg(src.is_neg());
      dst.copy_(src_cpu);
    }
  }

  if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
    iter.tensor(0).conj_physical_();
  }
  if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
    iter.tensor(0).neg_();
  }
}

at::Tensor& cnnl_copy_impl(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  TORCH_MLU_CHECK(self.defined(), "self is undefined");
  TORCH_MLU_CHECK(src.defined(), "src is undefined");
  TORCH_MLU_CHECK(
      !(self.is_sparse() || src.is_sparse()),
      "Sparse Tensor is not supported on MLU.");
  TORCH_MLU_CHECK(
      !(self.is_quantized() || src.is_quantized()),
      "Quantized Tensor is not supported on MLU.");
  TORCH_MLU_CHECK(
      (self.sizes() == src.sizes()) ||
          at::is_expandable_to(src.sizes(), self.sizes()),
      "src sizes should be equal to or be expandable to self sizes in copy_.");

  if (self.is_same(src)) {
    return self;
  }
  auto copy_src = src;
  if (self.sizes() != src.sizes()) {
    copy_src = src.expand(self.sizes());
  }
  auto iter = at::TensorIteratorConfig()
                  .add_output(self)
                  .add_input(copy_src)
                  .resize_outputs(false)
                  .check_all_same_dtype(false)
                  .check_all_same_device(false)
                  .build();
  if (iter.numel() == 0) {
    return self;
  }
  if (!self.is_complex() && src.is_complex()) {
    TORCH_WARN_ONCE(
        "Casting complex values to real discards the imaginary part");
  }
  copy_kernel_mlu(iter, non_blocking);
  return self;
}

at::Tensor& cnnl_copy_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking) {
  if (self._is_zerotensor()) {
    TORCH_MLU_CHECK(
        false,
        "ZeroTensors are immutable. Please materialize the tensor using `.clone()`, if you want a mutable zero tensor.");
  }
  if (src._is_zerotensor()) {
    return self.zero_();
  }
  cnnl_copy_impl(self, src, non_blocking);
  return self;
}

// _copy_from_and_resize is used for fallback and we only need copy from cpu to
// mlu
at::Tensor cnnl__copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(
      self.is_cpu() && dst.device().is_privateuseone(),
      "_copy_from_and_resize now only support copy from cpu tensor to mlu tensor");

  if (dst.numel() == 0) {
    dst.resize_as_(self);
  }
  cnnl_copy_(const_cast<at::Tensor&>(dst), self);
  return dst;
}

// _copy_from_and_resize_sparse is used for fallback and we only need copy from
// cpu to mlu
at::Tensor cnnl__copy_from_and_resize_sparse(
    const at::sparse::SparseTensor& self,
    const at::sparse::SparseTensor& dst) {
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(dst.is_sparse(), "dst is not SparseTensor");
  TORCH_CHECK(self.is_sparse(), "self is not SparseTensor");
  TORCH_CHECK(
      self.is_cpu() && dst.device().is_privateuseone(),
      "_copy_from_and_resize_sparse now only support copy from cpu tensor to mlu tensor");

  dst.resize_as_(self);
  dst._values().resize_as_(self._values());
  dst._indices().resize_as_(self._indices());
  at::Tensor cpu_values = self._values();
  at::Tensor cpu_indices = self._indices();
  at::Tensor mlu_values = dst._values();
  at::Tensor mlu_indices = dst._indices();
  cnnl__copy_from_and_resize(cpu_values, mlu_values);
  cnnl__copy_from_and_resize(cpu_indices, mlu_indices);
  dst._coalesced_(self.is_coalesced());
  return dst;
}

} // namespace ops
} // namespace torch_mlu
