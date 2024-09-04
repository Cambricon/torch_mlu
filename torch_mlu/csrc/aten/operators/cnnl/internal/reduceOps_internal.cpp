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

#include <vector>
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::vector<int64_t> getRealDim(
    const std::vector<int64_t>& input_dim,
    int64_t t_dim) {
  const int dim = input_dim.size();
  std::vector<int64_t> dim_vec(input_dim);
  // handle negative dims
  for (int64_t i = 0; i < dim; ++i) {
    if (dim_vec[i] < 0) {
      dim_vec[i] = dim_vec[i] + t_dim;
    }
  }
  // remove duplicate dims and sort them
  // e.g. (3,1,1) -> (1,3)
  std::set<int64_t> s(dim_vec.begin(), dim_vec.end());
  dim_vec.assign(s.begin(), s.end());
  dim_vec.resize(s.size());
  return dim_vec;
}

// Return 0,1,2,...,N-1 for all dims.
std::vector<int64_t> getAllDim(int64_t dim) {
  if (dim == 0)
    return std::vector<int64_t>{0};
  std::vector<int64_t> all_dims(dim, 0);
  for (int i = 1; i < dim; i++) {
    all_dims[i] = i;
  }
  return all_dims;
}

void cnnl_reduce_internal(
    const at::Tensor& input,
    at::Tensor& output,
    at::Tensor& index,
    const std::vector<int64_t>& reduce_axis,
    cnnlReduceOp_t reduce_mode,
    const cnnlReduceIndices_t reduce_indices,
    float norm_p) {
  /*
    cnnlReduceOps does not squeeze shape, no matter if keepdim is enabled or
    not. So desc_shpae is the same length as input.dim() with only reduced axis
    is 1, and output_shape is the expect shape of output due to keepdim.
  */
  std::vector<int64_t> reduce_dim =
      std::move(getRealDim(reduce_axis, input.dim()));
  if (reduce_dim.size() == 0) {
    reduce_dim = std::move(getAllDim(input.dim()));
  }

  // input
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto input_cnnl_dtype = getCnnlType(input_impl);
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);

  // index
  void* index_ptr = nullptr;
  uint32_t index_size_inbytes = 0;
  if (index.defined()) {
    auto index_impl = getMluTensorImpl(index);
    index_ptr = index_impl->mlu_data_ptr();
    index_size_inbytes = sizeof(int64_t) * index.numel();
  }

  // output
  void* output_ptr = nullptr;
  tensorDescPtr_t output_desc;
  if (output.defined()) {
    auto output_impl = getMluTensorImpl(output);
    output_ptr = output_impl->mlu_data_ptr();
    output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  }

  // TODO:(sifengyang)
  // by defualt, the bit width is promoted in CNNL_REDUCE_ADD,
  // CNNL_REDUCE_AVG, CNNL_REDUCE_NORM1, CNNL_REDUCE_NORM2 and CNNL_REDUCE_MUL,
  // and other cnnlReduceOp_t will not.
  auto tensor_type = getCnnlDataType(input.dtype());
  CnnlReduceDescriptor reduce_desc;
  reduce_desc.set(
      reduce_dim,
      reduce_mode,
      reduce_indices,
      CNNL_64BIT_INDICES,
      tensor_type,
      norm_p);

  size_t workspace_size = 0;
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(
      handle,
      input_desc.get(),
      output_desc.get(),
      reduce_desc.mut_desc(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlReduce(
      /* handle               */ handle,
      /* reduce_desc          */ reduce_desc.desc(),
      /* workspace            */ workspace_ptr.get(),
      /* workspace_size       */ workspace_size,
      /* alpha                */ alpha,
      /* input_desc           */ input_desc.get(),
      /* input                */ input_ptr,
      /* indices_size_inbytes */ index_size_inbytes,
      /* indices              */ index_ptr,
      /* beta                 */ beta,
      /* output_desc          */ output_desc.get(),
      /* output               */ output_ptr));
}

void cnnl_var_internal(
    const at::Tensor& self,
    at::Tensor& out,
    at::IntArrayRef dim,
    double correction_value) {
  // Currently cnnl only supports correction_value of bool type.
  bool unbiased = correction_value;
  int dim_size = dim.size();
  // init axis for cnnl kernel param, if dim_size is 0, means all dimensions is
  // reduced, equally axis is set to {0} meanwhile input tensor is set to 1-D.
  std::vector<int> axis(dim_size > 0 ? dim_size : 1, 0);
  for (int i = 0; i < dim_size; ++i) {
    axis[i] = at::maybe_wrap_dim(dim[i], self.dim());
  }
  // set CnnlStdVarMeanDescriptor
  CnnlStdVarMeanDescriptor stdvarmean_desc;
  stdvarmean_desc.set(axis, CNNL_VAR, unbiased);

  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  if (dim_size > 0) {
    input_desc.set(self);
    output_desc.set(out);
  } else {
    // if dim_size is 0, means all dimensions is reduced, equally set 1-D input
    // tensor for cnnl kernel
    input_desc.set_dim(self);
    output_desc.set_dim(out);
  }

  auto input_impl = getMluTensorImpl(self);
  auto input_ptr = input_impl->mlu_data_ptr();

  auto output_impl = getMluTensorImpl(out);
  auto output_ptr = output_impl->mlu_data_ptr();
  auto handle = getCurrentHandle();
  // get workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetStdVarMeanWorkspaceSize(
      handle, stdvarmean_desc.desc(), input_desc.desc(), &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlStdVarMean(
      handle,
      stdvarmean_desc.desc(),
      input_desc.desc(),
      input_ptr,
      ws_ptr.get(),
      workspace_size,
      NULL,
      NULL,
      output_desc.desc(),
      output_ptr,
      NULL,
      NULL));
}

void cnnl_std_internal(
    const at::Tensor& self,
    at::Tensor& out,
    at::IntArrayRef dim,
    double correction_value) {
  // Currently cnnl only supports correction_value of bool type.
  bool unbiased = correction_value;
  int dim_size = dim.size();
  // init axis for cnnl kernel param, if dim_size is 0, means all dimensions is
  // reduced, equally axis is set to {0} meanwhile input tensor is set to 1-D.
  std::vector<int> axis(dim_size > 0 ? dim_size : 1, 0);
  for (int i = 0; i < dim_size; ++i) {
    axis[i] = at::maybe_wrap_dim(dim[i], self.dim());
  }
  // set CnnlStdVarMeanDescriptor
  CnnlStdVarMeanDescriptor stdvarmean_desc;
  stdvarmean_desc.set(axis, CNNL_STD, unbiased);

  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor output_desc;
  if (dim_size > 0) {
    input_desc.set(self);
    output_desc.set(out);
  } else {
    // if dim_size is 0, means all dimensions is reduced, equally set 1-D input
    // tensor for cnnl kernel
    input_desc.set_dim(self);
    output_desc.set_dim(out);
  }

  auto input_impl = getMluTensorImpl(self);
  auto input_ptr = input_impl->mlu_data_ptr();

  auto output_impl = getMluTensorImpl(out);
  auto output_ptr = output_impl->mlu_data_ptr();
  auto handle = getCurrentHandle();
  // get workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetStdVarMeanWorkspaceSize(
      handle, stdvarmean_desc.desc(), input_desc.desc(), &workspace_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlStdVarMean(
      handle,
      stdvarmean_desc.desc(),
      input_desc.desc(),
      input_ptr,
      ws_ptr.get(),
      workspace_size,
      output_desc.desc(),
      output_ptr,
      NULL,
      NULL,
      NULL,
      NULL));
}

} // namespace ops
} // namespace torch_mlu
