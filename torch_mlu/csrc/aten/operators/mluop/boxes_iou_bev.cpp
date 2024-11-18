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

#include "aten/operators/mluop/mluop_kernel.h"
#include "aten/operators/mluop/internal/mluop_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor mluop_boxes_iou_bev(
    const at::Tensor& boxes_a,
    const at::Tensor& boxes_b) {
  // Input is [x, y, z, dx, dy, dz, heading],
  // Here we only use [x, y, dx, dy, heading], while z and dz is 0.

  TORCH_MLU_CHECK(boxes_a.dim() == 2, "Boxes_a is not a 2-dims Tensor.");
  TORCH_MLU_CHECK(boxes_b.dim() == 2, "Boxes_b is not a 2-dims Tensor.");
  TORCH_MLU_CHECK(
      boxes_a.size(1) == 7, "Boxes_a is not an [N, 7] shaped Tensor.");
  TORCH_MLU_CHECK(
      boxes_b.size(1) == 7, "Boxes_b is not an [N, 7] shaped Tensor.");
  TORCH_MLU_CHECK(
      boxes_a.scalar_type() == at::ScalarType::Float &&
          boxes_b.scalar_type() == at::ScalarType::Float,
      "inputs dtype should be float");

  std::vector<int64_t> shape{boxes_a.size(0), boxes_b.size(0)};
  auto output =
      at::empty(shape, boxes_a.options(), c10::MemoryFormat::Contiguous);

  if (boxes_a.numel() == 0 || boxes_b.numel() == 0) {
    return output;
  }

  auto boxes_a_xy = at::slice(boxes_a, 1, 0, 2, 1); // x,y a[0] - a[1]
  auto boxes_a_dxdy = at::slice(boxes_a, 1, 3, 5, 1); // dx,dy a[3] - a[4]
  auto boxes_a_h = at::slice(boxes_a, 1, 6, 7, 1); // head a[6]
  std::vector<at::Tensor> boxes_a_vec;
  boxes_a_vec.push_back(boxes_a_xy);
  boxes_a_vec.push_back(boxes_a_dxdy);
  boxes_a_vec.push_back(boxes_a_h);
  auto boxes_a_all = cnnl_contiguous(at::cat(at::TensorList(boxes_a_vec), 1));

  auto boxes_b_xy = at::slice(boxes_b, 1, 0, 2, 1); // x,y b[0] - b[1]
  auto boxes_b_dxdy = at::slice(boxes_b, 1, 3, 5, 1); // dx,dy b[3] - b[4]
  auto boxes_b_h = at::slice(boxes_b, 1, 6, 7, 1); // head b[6]
  std::vector<at::Tensor> boxes_b_vec;
  boxes_b_vec.push_back(boxes_b_xy);
  boxes_b_vec.push_back(boxes_b_dxdy);
  boxes_b_vec.push_back(boxes_b_h);
  auto boxes_b_all = cnnl_contiguous(at::cat(at::TensorList(boxes_b_vec), 1));

  mluop_boxes_iou_bev_internal(boxes_a_all, boxes_b_all, output);
  return output;
}

} // namespace ops
} // namespace torch_mlu
