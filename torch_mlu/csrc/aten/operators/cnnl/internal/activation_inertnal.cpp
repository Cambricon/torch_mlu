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
#include "ATen/NativeFunctions.h"
#include "aten/utils/dispatch.h"

namespace torch_mlu {
namespace ops {

static constexpr float SELU_ALPHA_MLU = 1.67326319217681884765625;
static constexpr float SELU_LAMBDA_MLU = 1.05070102214813232421875;

void set_activation_op_desc(
    const at::MemoryFormat& memory_format,
    CnnlActivationDescriptor& op_desc,
    cnnlActivationMode_t mode,
    /*only for glu*/ int64_t dim = 0,
    /*leaky-relu, elu, silu*/ const at::Scalar& negative_slope = 0.0,
    /*only for elu, silu*/ const at::Scalar& scale = 0.0,
    /*only for elu, silu*/ const at::Scalar& input_scale = 0.0,
    /*only for elu */ bool is_result = false,
    /*only for gelu*/ bool approximate = true) {
  cnnlActivationPreference_t prefer = CNNL_ACTIVATION_HIGH_PRECISION;
  if (mode == CNNL_ACTIVATION_GLU) {
    dim = modify_dim_based_on_layout(dim, memory_format);
    op_desc.set(mode, prefer, 0.0, dim, 0.0, 0.0, false, false);
  } else if (mode == CNNL_ACTIVATION_GELU) {
    op_desc.set(mode, prefer, 0.0, 0, 0.0, 0.0, false, approximate);
  } else if (
      mode == CNNL_ACTIVATION_LEAKYRELU || mode == CNNL_ACTIVATION_SOFTSHRINK ||
      mode == CNNL_ACTIVATION_HARDSHRINK) {
    float negative_slope_value = negative_slope.to<float>();
    op_desc.set(mode, prefer, negative_slope_value, 0, 0.0, 0.0, false, false);
  } else if (mode == CNNL_ACTIVATION_ELU_V2) {
    // according to gpu kernel, mlu side parameter is:
    // input_scale, scale, negative_slope -> coef, scale, gamma
    // and mlu only support float type.
    float coef_value = input_scale.to<float>();
    float gamma_value = negative_slope.to<float>();
    float scale_value = scale.to<float>();
    op_desc.set(
        mode,
        prefer,
        coef_value,
        0,
        gamma_value,
        scale_value,
        is_result,
        false);
  } else if (mode == CNNL_ACTIVATION_SELU) {
    op_desc.set(
        mode, prefer, 0.0, 0, SELU_ALPHA_MLU, SELU_LAMBDA_MLU, false, false);
  } else if (mode == CNNL_ACTIVATION_HARDSIGMOID) {
    op_desc.set(mode, prefer, 0.0, 0, 1 / 6.0, 0.5, false, false);
  } else {
    op_desc.set(mode, prefer, 0.0, 0, 0.0, 0.0, false, false);
  }
}

// 1) tanh, hardswish, hardsigmoid, logsigmoid, sigmoid,
//    silu without scalar and approximate;
// 2) glu with dim, and type need int;
// 3) gelu with approximate(bool) which is decision by user
//    elu, leaky-relu with scalars, and using opmath_type<xxx>
//    to get high precision type.
void cnnl_activation_internal(
    at::Tensor& output,
    const at::Tensor& input,
    cnnlActivationMode_t mode,
    /*only for glu*/ int64_t dim,
    /*leaky-relu, elu, silu*/ const at::Scalar& negative_slope,
    /*only for elu, silu*/ const at::Scalar& scale,
    /*only for elu, silu*/ const at::Scalar& input_scale,
    /*only for gelu*/ bool approximate) {
  if (output.numel() == 0)
    return;

  auto input_impl = getMluTensorImpl(input);

  auto input_ptr = mlu_data_ptr(input_impl);

  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto memory_format = input.suggest_memory_format();
  auto layout = suggestCnnlLayout(memory_format);
  auto input_desc = getTensorDesc(input_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);

  CnnlActivationDescriptor op_desc;
  set_activation_op_desc(
      memory_format,
      op_desc,
      mode,
      dim,
      negative_slope,
      scale,
      input_scale,
      false,
      approximate);

  // call cnnl activation interface
  auto handle = getCurrentHandle();
  const void* alpha = nullptr;
  const void* beta = nullptr;
  TORCH_CNNL_CHECK(cnnlActivationForward(
      /* handle          */ handle,
      /* activation_desc */ op_desc.desc(),
      /* alpha           */ alpha,
      /* x_desc          */ input_desc.get(),
      /* x               */ input_ptr,
      /* beta            */ beta,
      /* y_desc          */ output_desc.get(),
      /* y               */ output_ptr));
}

void cnnl_activation_backward_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& grad,
    cnnlActivationMode_t mode,
    /*only for glu*/ int64_t dim,
    /*leaky-relu, elu, silu*/ const at::Scalar& negative_slope,
    /*only for elu, silu*/ const at::Scalar& scale,
    /*only for elu, silu*/ const at::Scalar& input_scale,
    /*only for elu backward*/ bool is_result,
    /*only for gelu*/ bool approximate) {
  if (output.numel() == 0)
    return;

  auto grad_impl = getMluTensorImpl(grad);

  auto grad_ptr = mlu_data_ptr(grad_impl);

  auto self_impl = getMluTensorImpl(self);
  auto self_ptr = mlu_data_ptr(self_impl);

  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto memory_format = self.suggest_memory_format();
  auto layout = suggestCnnlLayout(memory_format);
  auto grad_desc = getTensorDesc(grad_impl, layout);
  auto self_desc = getTensorDesc(self_impl, layout);
  auto output_desc = getTensorDesc(output_impl, layout);

  CnnlActivationDescriptor op_desc;
  set_activation_op_desc(
      memory_format,
      op_desc,
      mode,
      dim,
      negative_slope,
      scale,
      input_scale,
      is_result,
      approximate);

  // call cnnl activation interface
  auto handle = getCurrentHandle();
  const void* alpha = nullptr;
  const void* beta = nullptr;
  if (mode == CNNL_ACTIVATION_SIGMOID || mode == CNNL_ACTIVATION_TANH) {
    TORCH_CNNL_CHECK(cnnlActivationBackward(
        /* handle          */ handle,
        /* activation_desc */ op_desc.desc(),
        /* alpha           */ alpha,
        /* y_desc          */ self_desc.get(),
        /* y               */ self_ptr,
        /* diff_y_desc     */ grad_desc.get(),
        /* diff_y          */ grad_ptr,
        /* x_desc          */ nullptr,
        /* x               */ nullptr,
        /* beta            */ beta,
        /* diff_x_desc     */ output_desc.get(),
        /* diff_x          */ output_ptr));
  } else {
    TORCH_CNNL_CHECK(cnnlActivationBackward(
        /* handle          */ handle,
        /* activation_desc */ op_desc.desc(),
        /* alpha           */ alpha,
        /* y_desc          */ nullptr,
        /* y               */ nullptr,
        /* diff_y_desc     */ grad_desc.get(),
        /* diff_y          */ grad_ptr,
        /* x_desc          */ self_desc.get(),
        /* x               */ self_ptr,
        /* beta            */ beta,
        /* diff_x_desc     */ output_desc.get(),
        /* diff_x          */ output_ptr));
  }
}

/***************************************softplus****************************************/
void cnnl_softplus_internal(
    at::Tensor& output,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar threshold) {
  const int beta_t = beta.to<int>();
  const int threshold_t = threshold.to<int>();

  auto [input_desc, input_ptr] = GetTensorDescAndMluDataPtr(input);
  auto [output_desc, output_ptr] = GetTensorDescAndMluDataPtr(output);

  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlSoftplusForward(
      /* handle  */ handle,
      /* x_desc  */ input_desc.get(),
      /* x       */ input_ptr,
      /* y_desc  */ output_desc.get(),
      /* y       */ output_ptr,
      /* beta    */ beta_t,
      /* threshold */ threshold_t));
}

void cnnl_softplus_backward_internal(
    const at::Tensor& self,
    const at::Tensor& grad_output,
    at::Scalar beta,
    at::Scalar threshold,
    at::Tensor& result) {
  const int beta_t = beta.to<int>();
  const int threshold_t = threshold.to<int>();
  // prepare cnnl self
  auto self_impl = getMluTensorImpl(self);

  auto self_ptr = mlu_data_ptr(self_impl);
  // prepare cnnl grad_input
  auto grad_input_impl = getMluTensorImpl(result);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);
  auto handle = getCurrentHandle();
  // prepare cnnl grad_output
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);
  auto cnnl_suggest_layout = suggestCnnlLayout(self_impl);
  auto grad_output_desc = getTensorDesc(grad_output_impl, cnnl_suggest_layout);
  auto self_desc = getTensorDesc(self_impl, cnnl_suggest_layout);
  auto grad_input_desc = getTensorDesc(grad_input_impl, cnnl_suggest_layout);

  TORCH_CNNL_CHECK(cnnlSoftplusBackward(
      /* handle      */ handle,
      /* x_desc      */ grad_output_desc.get(),
      /* x           */ grad_output_ptr,
      /* diff_y_desc */ self_desc.get(),
      /* diff_y      */ self_ptr,
      /* diff_x_desc */ grad_input_desc.get(),
      /* diff_x      */ grad_input_ptr,
      /* beta        */ beta_t,
      /* threshold   */ threshold_t));
}
/***************************************hardtanh****************************************/
void cnnl_hardtanh_backward_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& grad,
    at::Scalar min_val,
    at::Scalar max_val) {
  if (output.numel() == 0)
    return;
  const float min_f = min_val.to<float>();
  const float max_f = max_val.to<float>();
  // prepare cnnl input
  auto self_impl = getMluTensorImpl(self);

  auto self_ptr = mlu_data_ptr(self_impl);
  auto grad_impl = getMluTensorImpl(grad);
  auto grad_ptr = mlu_data_ptr(grad_impl);
  // prepare cnnl output
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto cnnl_suggest_layout = suggestCnnlLayout(self_impl);
  auto output_desc = getTensorDesc(output_impl, cnnl_suggest_layout);
  auto self_desc = getTensorDesc(self_impl, cnnl_suggest_layout);
  auto grad_desc = getTensorDesc(grad_impl, cnnl_suggest_layout);
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlHardtanhBackward(
      /* handle      */ handle,
      /* x_desc      */ self_desc.get(),
      /* x           */ self_ptr,
      /* diff_y_desc */ grad_desc.get(),
      /* diff_y      */ grad_ptr,
      /* max_val     */ max_f,
      /* min_val     */ min_f,
      /* diff_x_desc */ output_desc.get(),
      /* diff_x      */ output_ptr));
}
/***************************************threshold****************************************/
void cnnl_threshold_internal(
    at::Tensor& output,
    const at::Tensor& input,
    at::Scalar threshold,
    at::Scalar value) {
  if (output.numel() == 0)
    return;

  auto [input_desc, input_ptr] = GetTensorDescAndMluDataPtr(input);
  auto [output_desc, output_ptr] = GetTensorDescAndMluDataPtr(output);
  auto handle = getCurrentHandle();

  AT_DISPATCH_ALL_MLU_TYPES_AND_HALF_AND_BFLOAT16(
      input.scalar_type(), "cnnl_threshold", [&] {
        auto threshold_val = threshold.to<scalar_t>();
        auto value_val = value.to<scalar_t>();
        TORCH_CNNL_CHECK(cnnlThreshold(
            handle,
            input_desc.get(),
            input_ptr,
            static_cast<void*>(&threshold_val),
            static_cast<void*>(&value_val),
            output_desc.get(),
            output_ptr));
      });
}
void cnnl_threshold_backward_internal(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::Scalar threshold) {
  if (result.numel() == 0)
    return;
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto grad_input_impl = getMluTensorImpl(result);
  auto grad_output_impl = getMluTensorImpl(grad_output);
  auto cnnl_suggest_layout = suggestCnnlLayout(grad_output_impl);
  auto grad_input_desc = getTensorDesc(grad_input_impl, cnnl_suggest_layout);
  auto input_desc = getTensorDesc(input_impl, cnnl_suggest_layout);
  auto grad_output_desc = getTensorDesc(grad_output_impl, cnnl_suggest_layout);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);
  auto grad_output_ptr = mlu_data_ptr(grad_output_impl);
  // set descriptor config
  auto handle = getCurrentHandle();

  AT_DISPATCH_ALL_MLU_TYPES_AND_HALF_AND_BFLOAT16(
      input.scalar_type(), "cnnl_threshold_backward", [&] {
        auto threshold_val = threshold.to<scalar_t>();
        TORCH_CNNL_CHECK(cnnlThresholdBackward(
            handle,
            input_desc.get(),
            input_ptr,
            grad_output_desc.get(),
            grad_output_ptr,
            static_cast<void*>(&threshold_val),
            grad_input_desc.get(),
            grad_input_ptr));
      });
}

} // namespace ops
} // namespace torch_mlu
