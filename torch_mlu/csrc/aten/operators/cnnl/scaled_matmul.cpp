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

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "c10/core/ScalarType.h"
#include "scaled_matmul_utils.h"

namespace torch_mlu {
namespace ops {
namespace {

enum class ScalingType { TensorWise, RowWise, Error };
/*
 * Scaling Type Determination:
 * ---------------------------
 * Conditions and corresponding Scaling Types:
 *
 * - If scale_a.numel() == 1 && scale_b.numel() == 1:
 *   - Returns TensorWise.
 *
 * - Else if scale_a.dim() == 1 && scale_a.size(0) == dim_m && scale_b.size(0)
 * == dim_n:
 *   - Returns RowWise.
 *
 * - Otherwise:
 *   - Returns Error.
 */

// Validates the scale tensors to scaled_mm
// And returns the type of scaling/which kernel to use
ScalingType get_scaling_type(
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    int64_t dim_m,
    int64_t dim_n) {
  // Both Per-Tensor and Row-wise scaling expect fp32 tensors
  TORCH_CHECK(
      scale_a.scalar_type() == at::kFloat &&
          scale_b.scalar_type() == at::kFloat,
      "Both scale_a and scale_b must be float (fp32) tensors.");

  // Check the singluar scale case for per-tensor scaling
  if (scale_a.numel() == 1 && scale_b.numel() == 1) {
    return ScalingType::TensorWise;
  } else {
    // we currently not support non-Tensorwise scaling.
    return ScalingType::Error;
  }

  // For non-TensorWise scaling, enforce 2D input tensors
  TORCH_CHECK(
      scale_a.dim() == 2 && scale_b.dim() == 2,
      "For non-TensorWise scaling, scale tensors must be 2-dimensional, "
      "but got scale_a.dim()=",
      scale_a.dim(),
      " and scale_b.dim()=",
      scale_b.dim());

  // Check for RowWise scaling
  if (scale_a.size(0) == dim_m && scale_a.size(1) == 1 &&
      scale_b.size(0) == 1 && scale_b.size(1) == dim_n) {
    TORCH_CHECK(
        scale_a.is_contiguous() && scale_b.is_contiguous(),
        "Both scale_a and scale_b must be contiguous for RowWise scaling.");
    return ScalingType::RowWise;
  }

  // If we reach here, the input doesn't match any valid scaling type
  TORCH_CHECK(
      false,
      "Invalid scaling configuration. For TensorWise scaling, both scales should be scalar. "
      "For RowWise scaling, scale_a should be (",
      dim_m,
      ", 1) and scale_b should be (1, ",
      dim_n,
      "). "
      "Got scale_a.size()=(",
      scale_a.size(0),
      ", ",
      scale_a.size(1),
      ") and ",
      "scale_b.size()=(",
      scale_b.size(0),
      ", ",
      scale_b.size(1),
      ")");

  return ScalingType::Error;
}

at::Tensor getMMInput(const at::Tensor& self, const bool& trans) {
  if (trans) {
    return self.t();
  } else {
    return self;
  }
}

static bool _scaled_mm_allowed_device(const Tensor& self) {
  DeviceProp* prop = torch_mlu::getDeviceProperties(self.get_device());
  if (prop->major == 6) {
    return true;
  } else {
    return false;
  }
}
} // namespace

// Computes matrix multiply + bias while applying scaling to input and output
// matrices and computes amax Scales are only applicable when matrices are of
// Float8 type and assumbed to be equal to 1.0 by default. If output matrix type
// is 16 or 32-bit type, neither scale_result is applied nor amax is computed.
// Known limitations:
//  - Only works if mat1 is row-major and mat2 is column-major
//  - Only works if matrices sizes are divisible by 32
std::tuple<Tensor&, Tensor&> cnnl__scaled_mm_out(
    const Tensor& mat1,
    const Tensor& mat2,
    const c10::optional<at::Tensor>& bias,
    c10::optional<c10::ScalarType> out_dtype,
    const c10::optional<at::Tensor>& scale_a,
    const c10::optional<at::Tensor>& scale_b,
    const c10::optional<at::Tensor>& scale_result,
    Tensor& out,
    Tensor& amax) {
  // Check sizes
  bool allowed_device = _scaled_mm_allowed_device(mat1);
  // If scale_a/scale_b is not set, use default value 1.0
  auto scale_a_ = scale_a.value_or(at::ones(
      {}, c10::TensorOptions().device(at::kPrivateUse1).dtype(at::kFloat)));
  auto scale_b_ = scale_b.value_or(at::ones(
      {}, c10::TensorOptions().device(at::kPrivateUse1).dtype(at::kFloat)));
  TORCH_CHECK(
      allowed_device,
      "torch._scaled_mm is only supported on specific MLU series.");
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      " and ",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ")");
  TORCH_CHECK(
      scale_a_.numel() == 1 && scale_a_.scalar_type() == c10::kFloat,
      "scale_a must be float scalar");
  TORCH_CHECK(
      scale_b_.numel() == 1 && scale_b_.scalar_type() == c10::kFloat,
      "scale_b must be a float scalar");
  // Check what type of scaling we are doing based on inputs
  ScalingType scaling_choice =
      get_scaling_type(scale_a_, scale_b_, mat1.size(0), mat2.size(1));
  TORCH_INTERNAL_ASSERT(
      scaling_choice != ScalingType::Error, "Scaling type not supported");

  TORCH_CHECK(
      !scale_result ||
          (scale_result->numel() == 1 &&
           scale_result->scalar_type() == c10::kFloat),
      "scale_result must be a float scalar");
  TORCH_CHECK(
      !bias || bias->numel() == mat2.sizes()[1],
      "Bias must be size ",
      mat2.sizes()[1],
      " but got ",
      bias->numel());
  TORCH_CHECK(
      mat1.sizes()[1] % 16 == 0,
      "Expected trailing dimension of mat1 to be divisible by 16 ",
      "but got mat1 shape: (",
      mat1.sizes()[0],
      "x",
      mat1.sizes()[1],
      ").");
  TORCH_CHECK(
      mat2.sizes()[0] % 16 == 0 && mat2.sizes()[1] % 16 == 0,
      "mat2 shape (",
      mat2.sizes()[0],
      "x",
      mat2.sizes()[1],
      ") must be divisible by 16");
  // open this check once we support out_dtype
  // TORCH_CHECK(
  //    !out_dtype || *out_dtype == out.scalar_type(),
  //    "out_dtype must match output matrix type");
  TORCH_CHECK(
      isFloat8Type(mat1.scalar_type()),
      "Expected mat1 to be Float8 matrix got ",
      mat1.scalar_type());
  TORCH_CHECK(
      isFloat8Type(mat2.scalar_type()),
      "Expected mat2 to be Float8 matrix got ",
      mat2.scalar_type());
  if (bias) {
    TORCH_CHECK(
        bias->scalar_type() == out.scalar_type(),
        "Bias dtype should be same as out_dtype");
  }
  {
    auto bias_ = bias.value_or(Tensor());
    auto scale_result_ = scale_result.value_or(Tensor());

    // NOLINTNEXTLINE(*c-array*)
    at::TensorArg targs[]{
        {out, "out", 0},
        {mat1, "mat1", 1},
        {mat2, "mat2", 2},
        {bias_, "bias", 3},
        {scale_a_, "scale_a", 4},
        {scale_b_, "scale_b", 5},
        {scale_result_, "scale_result", 6}};
    checkAllSameMLU(__func__, targs);
  }
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  cnCommonArgs args(mat1, mat2, out);

  at::Tensor mata_tensor = *args.mata;
  at::Tensor matb_tensor = *args.matb;
  at::Tensor result_tensor = *args.out;
  mata_tensor = getMMInput(mata_tensor, (args.transa != args.trans_result));
  matb_tensor = getMMInput(matb_tensor, (args.transb != args.trans_result));
  result_tensor = getMMInput(result_tensor, args.trans_result);

  // Currently we only support scale_a/scale_b to be scalar
  at::Scalar scale_a_scalar = scale_a_.item();
  at::Scalar scale_b_scalar = scale_b_.item();

  if (bias.has_value()) {
    cnnl_scaled_mm_bias_out_internal(
        result_tensor,
        mata_tensor,
        matb_tensor,
        args.transa,
        args.transb,
        scale_a_scalar,
        scale_b_scalar,
        bias.value());
  } else {
    cnnl_scaled_mm_out_internal(
        result_tensor,
        mata_tensor,
        matb_tensor,
        args.transa,
        args.transb,
        scale_a_scalar,
        scale_b_scalar);
  }

  if (!out.is_same(result_tensor)) {
    out.copy_(result_tensor);
  }

  // Currently, we don't support amax, calculate separately
  amax = at::max(at::abs(out.to(c10::kFloat)));

  return {out, amax};
}

std::tuple<Tensor, Tensor> cnnl__scaled_mm(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const c10::optional<at::Tensor>& bias,
    c10::optional<c10::ScalarType> out_dtype,
    const c10::optional<at::Tensor>& scale_a,
    const c10::optional<at::Tensor>& scale_b,
    const c10::optional<at::Tensor>& scale_result) {
  // currently, we only support out_dtype to be float/half/bfloat16, remove this
  // limit once supported. const auto out_dtype_ =
  // out_dtype.value_or(mat_a.scalar_type());
  Tensor out;
  if (out_dtype.has_value() &&
      (*out_dtype == c10::ScalarType::Float ||
       *out_dtype == c10::ScalarType::Half ||
       *out_dtype == c10::ScalarType::BFloat16)) {
    out = at::empty({0}, mat_a.options().dtype(*out_dtype));
  } else {
    const auto out_dtype_ = c10::ScalarType::Float;
    out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  }
  Tensor amax = at::empty({0}, mat_a.options().dtype(c10::ScalarType::Float));
  auto [result, amax_result] = cnnl__scaled_mm_out(
      mat_a,
      mat_b,
      bias,
      out_dtype,
      scale_a,
      scale_b,
      scale_result,
      out,
      amax);

  // Currently, cnnl not support scale result, we do this operation here
  if (scale_result.has_value()) {
    at::Scalar scale_result_ = scale_result->item();
    at::native::mul_(result, scale_result_);
  }
  // As we always process out dtype as float, we need to cast output back to
  // the original out_dtype. Delete after out_dtype param is supported.
  if (!out_dtype) {
    out_dtype = mat_a.scalar_type();
  }
  if (out_dtype != result.scalar_type()) {
    return {result.to(out_dtype.value()), amax};
  }
  return {result, amax};
}

} //  namespace ops
} //  namespace torch_mlu
