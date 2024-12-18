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

namespace torch_mlu {
namespace ops {
void cnnl_dynamic_partition_internal(
    at::Tensor& out_counts,
    at::Tensor& output,
    const at::Tensor& data,
    const at::Tensor& partitions,
    const int num_partitions) {
  // get tensor desc and ptr
  auto data_impl = getMluTensorImpl(data);
  auto data_desc = getTensorDesc(data_impl, CNNL_LAYOUT_ARRAY);
  auto data_ptr = data_impl->mlu_data_ptr();
  auto partitions_impl = getMluTensorImpl(partitions);
  auto partitions_desc = getTensorDesc(partitions_impl, CNNL_LAYOUT_ARRAY);
  auto partitions_ptr = partitions_impl->mlu_data_ptr();
  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl);
  auto output_ptr = output_impl->mlu_data_ptr();
  auto out_counts_impl = getMluTensorImpl(out_counts);
  auto out_counts_desc = getTensorDesc(out_counts_impl);
  auto out_counts_ptr = out_counts_impl->mlu_data_ptr();

  // get workspace
  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetDynamicPartitionWorkspaceSize(
      handle,
      data_desc.get(),
      partitions_desc.get(),
      num_partitions,
      &workspace_size));
  at::DataPtr workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlDynamicPartition(
      handle,
      data_desc.get(),
      data_ptr,
      partitions_desc.get(),
      partitions_ptr,
      num_partitions,
      workspace_ptr.get(),
      workspace_size,
      output_desc.get(),
      output_ptr,
      out_counts_desc.get(),
      out_counts_ptr));
  return;
}

} // namespace ops
} // namespace torch_mlu
