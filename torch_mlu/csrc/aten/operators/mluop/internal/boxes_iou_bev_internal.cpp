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

#include "aten/operators/mluop/internal/mluop_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& mluop_boxes_iou_bev_internal(
    const at::Tensor& boxes_a,
    const at::Tensor& boxes_b,
    at::Tensor& ans_iou) {
  /*
  skip test for net check only.
  */
  /* assert output shape is A*B*/
  auto a_impl = getMluTensorImpl(boxes_a);
  auto a_ptr = mlu_data_ptr(a_impl);

  auto b_impl = getMluTensorImpl(boxes_b);
  auto b_ptr = mlu_data_ptr(b_impl);

  auto ans_impl = getMluTensorImpl(ans_iou);
  auto ans_ptr = mlu_data_ptr(ans_impl);

  MluOpTensorDescriptor a_desc;
  MluOpTensorDescriptor b_desc;
  MluOpTensorDescriptor ans_desc;

  a_desc.set(boxes_a);
  b_desc.set(boxes_b);
  ans_desc.set(ans_iou);

  int mode = 0; // IOU mode.
  bool aligned = false; // A x B mode.

  auto handle = getCurrentMluOpHandle();

  TORCH_MLUOP_CHECK(mluOpBoxIouRotated(
      handle,
      mode,
      aligned,
      a_desc.desc(),
      a_ptr,
      b_desc.desc(),
      b_ptr,
      ans_desc.desc(),
      ans_ptr));
  return ans_iou;
}

} // namespace ops
} // namespace torch_mlu
