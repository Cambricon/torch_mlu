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

at::Tensor& cnnl_convolution_forward_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int64_t* padding,
    const int64_t* stride,
    const int64_t* dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    bool is_depth_wise_conv) {
  auto input_impl = getMluTensorImpl(input);
  auto weight_impl = getMluTensorImpl(weight);
  auto output_impl = getMluTensorImpl(output);
  tensorDescPtr_t input_desc;
  tensorDescPtr_t weight_desc;
  tensorDescPtr_t bias_desc;
  tensorDescPtr_t output_desc;
  CnnlConvolutionDescriptor conv_desc;
  size_t workspace_size = 0;
  // get current handle
  auto handle = getCurrentHandle();

  // prepare desc
  const int64_t input_dim = input.dim();
  cnnlTensorLayout_t layout =
      input_dim > 4 ? CNNL_LAYOUT_NDHWC : CNNL_LAYOUT_NHWC;
  auto output_cnnl_type = getCnnlDataType(output.scalar_type());
  auto input_scalar_type = input.scalar_type();
  auto input_cnnl_type = getCnnlDataType(input_scalar_type);
  auto weight_cnnl_type = getCnnlDataType(weight.scalar_type());
  const bool promote_compute_dtype =
      (output_cnnl_type == CNNL_DTYPE_HALF ||
       output_cnnl_type == CNNL_DTYPE_BFLOAT16);
  auto compute_dtype =
      promote_compute_dtype ? CNNL_DTYPE_FLOAT : output_cnnl_type;
  // Modify allow_tf32 based on input tensor scalar type
  if (input_scalar_type != at::kFloat)
    allow_tf32 = false;
  const auto& weight_size = weight.sizes();
  if (is_can_coalesce_second_dim(
          weight_size, input_dim, padding, stride, dilation)) {
    constexpr int64_t fixed_dim = 4;
    std::vector<int64_t> tensor_shape;
    tensor_shape.resize(fixed_dim);
    coalesce_conv_second_dim(
        input, input_cnnl_type, CNNL_DTYPE_INVALID, input_desc, tensor_shape);
    coalesce_conv_second_dim(
        weight,
        weight_cnnl_type,
        CNNL_DTYPE_INVALID,
        weight_desc,
        tensor_shape);
    coalesce_conv_second_dim(
        output, output_cnnl_type, compute_dtype, output_desc, tensor_shape);
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
        input, input_cnnl_type, CNNL_DTYPE_INVALID, input_desc, tensor_shape);
    coalesce_conv_last_dim(
        weight,
        weight_cnnl_type,
        CNNL_DTYPE_INVALID,
        weight_desc,
        tensor_shape);
    coalesce_conv_last_dim(
        output, output_cnnl_type, compute_dtype, output_desc, tensor_shape);
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
    input_desc = getTensorDesc(input_impl, input_cnnl_type, layout);
    // depth wise only support 4 dimension.
    if (is_depth_wise_conv) {
      weight_desc =
          getTensorDesc(weight_impl, weight_cnnl_type, CNNL_LAYOUT_HWCN);
    } else {
      weight_desc = getTensorDesc(weight_impl, weight_cnnl_type, layout);
    }
    output_desc =
        getTensorDesc(output_impl, output_cnnl_type, layout, compute_dtype);
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
  cnnlConvolutionFwdPreference_t pre_t = CNNL_CONVOLUTION_FWD_FASTEST;
  cnnlConvolutionForwardAlgo_t algo_t;

  TORCH_CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(
      handle,
      conv_desc.desc(),
      input_desc.get(),
      weight_desc.get(),
      output_desc.get(),
      pre_t,
      &algo_t));

  // prepare bias
  void* bias_ptr = nullptr;
  if (bias.defined() && bias.dim() != 0 && bias.numel() != 0) {
    TORCH_CHECK(
        bias.dim() == 1,
        "currently only support 1-dim bias in "
        "cnnl_float_convolution_internal when bias.dim() != 0, but got ",
        bias.dim(),
        " dim.");
    auto bias_impl = getMluTensorImpl(bias);
    bias_desc = getTensorDesc(bias_impl, CNNL_LAYOUT_ARRAY);
    bias_ptr = mlu_data_ptr(bias_impl);
  }

  // prepare workspace
  TORCH_CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
      handle,
      input_desc.get(),
      weight_desc.get(),
      output_desc.get(),
      bias_desc.get(),
      conv_desc.desc(),
      algo_t,
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto weight_ptr = mlu_data_ptr(weight_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  const void* alpha = nullptr;
  const void* beta = nullptr;

  TORCH_CNNL_CHECK(cnnlConvolutionForward(
      /* handle         */ handle,
      /* conv_desc      */ conv_desc.desc(),
      /* algo           */ algo_t,
      /* alpha          */ alpha,
      /* x_desc         */ input_desc.get(),
      /* x_ptr          */ input_ptr,
      /* w_desc         */ weight_desc.get(),
      /* w_ptr          */ weight_ptr,
      /* bias_desc      */ bias_desc.get(),
      /* bias_ptr       */ bias_ptr,
      /* workspace      */ workspace_ptr.get(),
      /* workspace_size */ workspace_size,
      /* beta           */ beta,
      /* y_desc         */ output_desc.get(),
      /* y_ptr          */ output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
