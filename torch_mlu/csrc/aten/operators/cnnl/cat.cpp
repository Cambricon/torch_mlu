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

#include <ATen/native/TensorShape.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/MemoryOverlap.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

TORCH_IMPL_FUNC(cat_out_mlu)
(const at::ITensorListRef& tensors,
 int64_t dim,
 int64_t valid,
 bool all_contiguous,
 bool all_same_dtype,
 bool all_same_sizes_and_stride,
 c10::MemoryFormat memory_format,
 const at::Tensor& result) {
  if (result.numel() == 0) {
    return;
  }

  // Debug info
  // LOG(INFO) << "size " << tensors.size()
  //           << " all_contiguous " << all_contiguous
  //           << " all_same_dtype " << all_same_dtype
  //           << " ass " << all_same_sizes_and_stride
  //           << " memory_format " << memory_format
  //           << " result is cont " << result.is_contiguous(memory_format);

  // Fast path
  if C10_LIKELY (all_contiguous && all_same_dtype) {
    cnnl_cat_internal(tensors, result, dim, memory_format);
    return;
  }

  // Slow path
  auto materialized = tensors.materialize();
  auto input_size = materialized.size();
  c10::SmallVector<at::Tensor, 16> new_tensors;
  new_tensors.reserve(input_size);

  c10::ScalarType output_dtype = result.scalar_type();

  for (const auto i : c10::irange(input_size)) {
    const at::Tensor& t = materialized[i];

    // Must skip legacy zero dim tensors because
    // they are illegal to CNNL op.
    if (at::native::cat_should_skip_tensor(t)) {
      continue;
    }
    bool need_convert_dtype = false;
    bool need_contiguous = false;

    if (all_contiguous == false && !t.is_contiguous(memory_format)) {
      need_contiguous = true;
    }

    if (all_same_dtype == false && t.scalar_type() != output_dtype) {
      need_convert_dtype = true;
    }

    if (need_convert_dtype && need_contiguous) {
      const auto& temp = convertTensorType(t, output_dtype);
      new_tensors.emplace_back(cnnl_contiguous(temp, memory_format));
    } else if (need_convert_dtype) {
      new_tensors.emplace_back(convertTensorType(t, output_dtype));
    } else if (need_contiguous) {
      new_tensors.emplace_back(cnnl_contiguous(t, memory_format));
    } else {
      new_tensors.emplace_back(t);
    }
  }

  at::ITensorListRef new_tensors_list_ref(new_tensors);

  if (result.is_contiguous(memory_format)) {
    cnnl_cat_internal(new_tensors_list_ref, result, dim, memory_format);
  } else {
    // result is defined and has a different memory format
    auto new_result = at::empty_like(result, memory_format);
    cnnl_cat_internal(new_tensors_list_ref, new_result, dim, memory_format);
    result.copy_(new_result);
  }
}

} // namespace ops
} // namespace torch_mlu
