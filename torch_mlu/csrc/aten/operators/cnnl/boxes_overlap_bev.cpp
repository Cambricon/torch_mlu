/*
All modification made by Cambricon Corporation: © 2023 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2023, the respective contributors
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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

namespace {
static const int magic_num = 7;
}

at::Tensor& cnnl_boxes_overlap_bev_out(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other) {
  TORCH_MLU_CHECK(
      self.dim() == 2 && other.dim() == 2, "Input tensor dims only support 2.");
  TORCH_MLU_CHECK(
      self.size(0) != 0 && other.size(0) != 0,
      "The first dim size of input tensors need be greater than 0.");
  TORCH_MLU_CHECK(
      self.size(1) == magic_num && other.size(1) == magic_num,
      "The second dim size of input tensors need be ",
      magic_num,
      ".");
  TORCH_MLU_CHECK(
      self.scalar_type() == at::kFloat && other.scalar_type() == at::kFloat,
      "Only support float dtype.");
  // intput tensors contiguous
  auto self_contiguous = cnnl_contiguous(self);
  auto other_contiguous = cnnl_contiguous(other);
  // Output tensor
  std::vector<int64_t> result_shape{self.size(0), other.size(0)};
  if (!output.defined()) {
    output = at::empty(
        result_shape, self_contiguous.options(), c10::MemoryFormat::Contiguous);
  }
  TORCH_MLU_CHECK(
      self.scalar_type() == output.scalar_type(),
      "Input dtype and output dtype need be same.");
  if (!output.sizes().equals(result_shape)) {
    output.resize_(result_shape);
  }
  // output tensor is not contiguous
  auto output_contiguous = output.is_contiguous()
      ? output
      : at::empty(
            result_shape,
            self_contiguous.options(),
            c10::MemoryFormat::Contiguous);
  cnnl_boxes_overlap_bev_out_internal(
      output_contiguous, self_contiguous, other_contiguous);
  // Output tensor is not contiguous.
  if (!output_contiguous.is_same(output)) {
    output.copy_(output_contiguous);
  }
  return output;
}

at::Tensor cnnl_boxes_overlap_bev(
    const at::Tensor& self,
    const at::Tensor& other) {
  at::Tensor result;
  cnnl_boxes_overlap_bev_out(result, self, other);
  return result;
}

} // end of namespace ops
} // end of namespace torch_mlu
