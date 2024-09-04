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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/internal_util.h"

namespace torch_mlu {
namespace ops {

void cnnl_softmax_out_internal(
    const at::Tensor& input,
    int64_t dim,
    at::Tensor& output,
    cnnlSoftmaxAlgorithm_t algo) {
  const int64_t ndim = input.dim();
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  auto layout = suggestCnnlLayout(input.suggest_memory_format());
  dim = torch_mlu::modify_dim_based_on_layout(dim, layout);
  auto input_desc =
      getTensorDescWithKeepDimAndOnlyThreeDims(input_impl, dim, layout);
  auto output_desc =
      getTensorDescWithKeepDimAndOnlyThreeDims(output_impl, dim, layout);
  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  // set descriptor config
  cnnlSoftmaxMode_t mode;
  if (ndim == 0 || dim == (ndim - 1)) {
    mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  } else if (dim == 0) {
    mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
  } else {
    mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
  }
  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlSoftmaxForward(
      /* handle    */ handle,
      /* algorithm */ algo,
      /* mode      */ mode,
      /* alpha     */ alpha,
      /* x_desc    */ input_desc.get(),
      /* x         */ input_ptr,
      /* beta      */ beta,
      /* y_desc    */ output_desc.get(),
      /* y         */ output_ptr));
}

void cnnl_softmax_backward_out_internal(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::Tensor& grad_input,
    cnnlSoftmaxAlgorithm_t algo) {
  const int64_t ndim = grad_output.dim();
  auto diff_y_impl = getMluTensorImpl(grad_output);
  auto y_impl = getMluTensorImpl(output);
  auto diff_x_impl = getMluTensorImpl(grad_input);
  // get current handle
  auto handle = getCurrentHandle();
  auto layout = suggestCnnlLayout(grad_output.suggest_memory_format());
  dim = torch_mlu::modify_dim_based_on_layout(dim, layout);
  auto diff_y_desc =
      getTensorDescWithKeepDimAndOnlyThreeDims(diff_y_impl, dim, layout);
  auto y_desc = getTensorDescWithKeepDimAndOnlyThreeDims(y_impl, dim, layout);
  auto diff_x_desc =
      getTensorDescWithKeepDimAndOnlyThreeDims(diff_x_impl, dim, layout);
  // malloc mlu memory
  auto diff_y_ptr = mlu_data_ptr(diff_y_impl);
  auto y_ptr = mlu_data_ptr(y_impl);
  auto diff_x_ptr = mlu_data_ptr(diff_x_impl);
  // set descriptor config
  cnnlSoftmaxMode_t mode;
  if (ndim == 0 || dim == (ndim - 1)) {
    mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  } else if (dim == 0) {
    mode = CNNL_SOFTMAX_MODE_HIGH_DIMENSION;
  } else {
    mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
  }
  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlSoftmaxBackward(
      /* handle      */ handle,
      /* algorithm   */ algo,
      /* mode        */ mode,
      /* alpha       */ alpha,
      /* y_desc      */ y_desc.get(),
      /* y           */ y_ptr,
      /* diff_y_desc */ diff_y_desc.get(),
      /* diif_y      */ diff_y_ptr,
      /* beta        */ beta,
      /* diff_x_desc */ diff_x_desc.get(),
      /* diff _x     */ diff_x_ptr));
}

} // namespace ops
} // namespace torch_mlu
