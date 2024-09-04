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
void cnnl_svd_internal(
    std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>& outputs,
    const at::Tensor& self,
    bool some,
    bool compute_uv,
    const at::Tensor& infos) {
  at::Tensor U, S, V;
  std::tie(U, S, V) = outputs;

  auto input_impl = getMluTensorImpl(self);
  auto input_ptr = mlu_data_ptr(input_impl);
  CnnlTensorDescriptor desc_input;
  desc_input.set(self);

  auto u_impl = getMluTensorImpl(U);
  auto u_ptr = mlu_data_ptr(u_impl);
  CnnlTensorDescriptor desc_u;
  desc_u.set(U);

  auto s_impl = getMluTensorImpl(S);
  auto s_ptr = mlu_data_ptr(s_impl);
  CnnlTensorDescriptor desc_s;
  desc_s.set(S);

  auto v_impl = getMluTensorImpl(V);
  auto v_ptr = mlu_data_ptr(v_impl);
  CnnlTensorDescriptor desc_v;
  desc_v.set(V);

  auto infos_impl = getMluTensorImpl(infos);
  auto infos_ptr = mlu_data_ptr(infos_impl);
  CnnlTensorDescriptor desc_infos;
  desc_infos.set(infos);

  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetSvdWorkspaceSize(
      handle, desc_input.desc(), some, compute_uv, &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  bool is_trans_u = false, is_trans_v = true;
  TORCH_CNNL_CHECK(cnnlSvd(
      handle,
      desc_input.desc(),
      input_ptr,
      some,
      compute_uv,
      is_trans_u,
      is_trans_v,
      workspace_size,
      workspace_ptr.get(),
      desc_u.desc(),
      u_ptr,
      desc_s.desc(),
      s_ptr,
      desc_v.desc(),
      v_ptr,
      desc_infos.desc(),
      infos_ptr));
}

} // namespace ops
} // namespace torch_mlu
