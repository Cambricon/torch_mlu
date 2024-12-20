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

#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/utils.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_nonzero_internal(at::Tensor& out, const at::Tensor& self) {
  // this is a layout sensitive op, but cnnl does not support strides, so we
  // need transpose
  auto dim_num = self.dim();
  auto handle = getCurrentHandle();
  auto self_impl = getMluTensorImpl(self);
  auto self_desc = getTensorDesc(self_impl);
  auto self_ptr = mlu_data_ptr(self_impl);

  // call cnnlNumTrue to count the number of nonzero elements
  at::Tensor num_true = at::empty(1, self.options().dtype(at::ScalarType::Int));
  auto num_true_impl = getMluTensorImpl(num_true);
  auto num_true_desc = getTensorDesc(num_true_impl);
  auto num_true_ptr = mlu_data_ptr(num_true_impl);

  size_t num_true_workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetNumTrueWorkspaceSize_v2(
      handle, num_true_desc.get(), &num_true_workspace_size));
  auto num_true_workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(num_true_workspace_size);
  TORCH_CNNL_CHECK(cnnlNumTrue_v3(
      handle,
      self_desc.get(),
      self_ptr,
      num_true_workspace_ptr.get(),
      num_true_workspace_size,
      num_true_desc.get(),
      num_true_ptr));
  size_t where_workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetWhereWorkspaceSize(
      handle, num_true_desc.get(), &where_workspace_size));
  auto where_workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(where_workspace_size);

  // call cnnlWhere to output the index of nonzero elements
  auto stream = getCurrentMLUStream();
  uint32_t num_nonzeros = 0;
  TORCH_CNRT_CHECK(cnrtMemcpyAsync_V2(
      &num_nonzeros,
      num_true_ptr,
      sizeof(uint32_t),
      stream.stream(),
      cnrtMemcpyDevToHost));
  stream.synchronize();
  const c10::SmallVector<int64_t, 2> outshape = {num_nonzeros, dim_num};

  if (!out.sizes().equals(outshape)) {
    out.resize_(outshape);
  }

  if (dim_num == 0) { // support scalar input
    return out;
  }
  auto out_impl = getMluTensorImpl(out);
  auto out_desc = getTensorDesc(out_impl);
  auto out_ptr = mlu_data_ptr(out_impl);

  TORCH_CNNL_CHECK(cnnlWhere_v2(
      handle,
      self_desc.get(),
      self_ptr,
      num_true_desc.get(),
      num_true_ptr,
      false,
      where_workspace_ptr.get(),
      where_workspace_size,
      out_desc.get(),
      static_cast<int*>(out_ptr)));
  return out;
}

} // namespace ops
} // namespace torch_mlu
