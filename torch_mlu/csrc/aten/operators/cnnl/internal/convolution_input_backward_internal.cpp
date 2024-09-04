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

#include "aten/operators/cnnl/internal/convolution_internal_utils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// Get poor performance when using HWCN in grad weight, and even in depth-wise.
// So still using NHWC in grad weight.
at::Tensor& cnnl_convolution_backward_input_internal(
    at::Tensor& input_grad,
    const at::Tensor& output_grad,
    const at::Tensor& weight,
    const int64_t* stride,
    const int64_t* padding,
    const int64_t* dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  auto input_grad_impl = getMluTensorImpl(input_grad);
  auto weight_impl = getMluTensorImpl(weight);
  auto output_grad_impl = getMluTensorImpl(output_grad);
  tensorDescPtr_t input_grad_desc;
  tensorDescPtr_t weight_desc;
  tensorDescPtr_t output_grad_desc;
  CnnlConvolutionDescriptor conv_desc;
  // get current handle
  auto handle = getCurrentHandle();

  // prepare desc
  const int64_t input_dim = output_grad.dim();
  auto layout = input_dim > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  auto input_grad_cnnl_type = getCnnlDataType(input_grad.scalar_type());
  auto weight_cnnl_type = getCnnlDataType(weight.scalar_type());
  auto output_grad_scalar_type = output_grad.scalar_type();
  auto output_grad_cnnl_type = getCnnlDataType(output_grad_scalar_type);
  // TODO(CNNLCORE-16695): Check failed in cnnl side, add this later.
  // const bool promote_compute_dtype = (input_grad_cnnl_type == CNNL_DTYPE_HALF
  // ||
  //                                     input_grad_cnnl_type ==
  //                                     CNNL_DTYPE_BFLOAT16);
  const bool promote_compute_dtype =
      input_grad_cnnl_type == CNNL_DTYPE_BFLOAT16;
  auto compute_dtype =
      promote_compute_dtype ? CNNL_DTYPE_FLOAT : input_grad_cnnl_type;
  // Modify allow_tf32 based on input tensor dtype
  if (output_grad_scalar_type != at::kFloat)
    allow_tf32 = false;
  const auto& weight_size = weight.sizes();
  if (is_can_coalesce_second_dim(
          weight_size, input_dim, padding, stride, dilation)) {
    constexpr int64_t fixed_dim = 4;
    std::vector<int64_t> tensor_shape;
    tensor_shape.resize(fixed_dim);
    coalesce_conv_second_dim(
        input_grad,
        input_grad_cnnl_type,
        compute_dtype,
        input_grad_desc,
        tensor_shape);
    coalesce_conv_second_dim(
        weight,
        weight_cnnl_type,
        CNNL_DTYPE_INVALID,
        weight_desc,
        tensor_shape);
    coalesce_conv_second_dim(
        output_grad,
        output_grad_cnnl_type,
        CNNL_DTYPE_INVALID,
        output_grad_desc,
        tensor_shape);
    int64_t padding_t[2] = {padding[1], padding[2]};
    int64_t stride_t[2] = {stride[1], stride[2]};
    int64_t dilation_t[2] = {dilation[1], dilation[2]};
    conv_desc.set(
        fixed_dim,
        stride_t,
        padding_t,
        dilation_t,
        groups,
        compute_dtype,
        allow_tf32);
  } else if (is_can_coalesce_last_dim(
                 weight_size, input_dim, padding, stride, dilation)) {
    constexpr int64_t fixed_dim = 4;
    std::vector<int64_t> tensor_shape;
    tensor_shape.resize(fixed_dim);
    coalesce_conv_last_dim(
        input_grad,
        input_grad_cnnl_type,
        compute_dtype,
        input_grad_desc,
        tensor_shape);
    coalesce_conv_last_dim(
        weight,
        weight_cnnl_type,
        CNNL_DTYPE_INVALID,
        weight_desc,
        tensor_shape);
    coalesce_conv_last_dim(
        output_grad,
        output_grad_cnnl_type,
        CNNL_DTYPE_INVALID,
        output_grad_desc,
        tensor_shape);
    int64_t padding_t[2] = {padding[0], 0};
    int64_t stride_t[2] = {stride[0], 1};
    int64_t dilation_t[2] = {dilation[0], 1};
    conv_desc.set(
        fixed_dim,
        stride_t,
        padding_t,
        dilation_t,
        groups,
        compute_dtype,
        allow_tf32);
  } else {
    input_grad_desc = getTensorDesc(
        input_grad_impl, input_grad_cnnl_type, layout, compute_dtype);
    weight_desc = getTensorDesc(weight_impl, weight_cnnl_type, layout);
    output_grad_desc =
        getTensorDesc(output_grad_impl, output_grad_cnnl_type, layout);
    conv_desc.set(
        input_dim,
        stride,
        padding,
        dilation,
        groups,
        compute_dtype,
        allow_tf32);
  }
  // prepare conv desc
  cnnlConvolutionBwdDataPreference_t pre_t = CNNL_CONVOLUTION_BWD_DATA_FASTEST;
  cnnlConvolutionBwdDataAlgo_t algo_t;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardDataAlgorithm(
      handle,
      weight_desc.get(),
      output_grad_desc.get(),
      conv_desc.desc(),
      input_grad_desc.get(),
      pre_t,
      &algo_t));
  // prepare workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetConvolutionBackwardDataWorkspaceSize(
      handle,
      weight_desc.get(),
      output_grad_desc.get(),
      conv_desc.desc(),
      input_grad_desc.get(),
      algo_t,
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  // malloc mlu memory
  auto grad_input_ptr = input_grad_impl->mlu_data_ptr();
  auto weight_ptr = weight_impl->mlu_data_ptr();
  auto grad_ptr = output_grad_impl->mlu_data_ptr();
  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlConvolutionBackwardData(
      /* handle         */ handle,
      /* alpha          */ alpha,
      /* weight_desc    */ weight_desc.get(),
      /* weight         */ weight_ptr,
      /* diff_y_desc    */ output_grad_desc.get(),
      /* diff_y         */ grad_ptr,
      /* conv_desc      */ conv_desc.desc(),
      /* algo           */ algo_t,
      /* workspace      */ workspace_ptr.get(),
      /* workspace_size */ workspace_size,
      /* beta           */ beta,
      /* diff_x_desc    */ input_grad_desc.get(),
      /* diff_x         */ grad_input_ptr));
  return input_grad;
}

} // namespace ops
} // namespace torch_mlu
