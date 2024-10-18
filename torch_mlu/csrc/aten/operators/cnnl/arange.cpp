#include <c10/util/Optional.h>
#include "aten/utils/dispatch.h"
#include "ATen/InferSize.h"
#include "ATen/NativeFunctions.h"
#include "ATen/AccumulateType.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// CNNL only support:
// start, step, output
// int32, int32, int32
// float, float, float
// float, float, half
// float, float, bfloat16
at::Tensor& cnnl_arange_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {
  // complex type not supported
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      out.scalar_type(),
      "arange_mlu",
      [&]() {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(
            std::isfinite(static_cast<double>(xstart)) &&
                std::isfinite(static_cast<double>(xend)),
            "unsupported range: ",
            xstart,
            " -> ",
            xend);
        TORCH_CHECK(
            ((xstep > 0) && (xend >= xstart)) ||
                ((xstep < 0) && (xend <= xstart)),
            "upper bound and larger bound inconsistent with step sign");

        // the corner-case we do want to take into account is int64_t,
        // which has higher precision than double
        double size_d;
        if (std::is_same<scalar_t, int64_t>::value) {
          int64_t sgn = (xstep > 0) - (xstep < 0);
          size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
        } else {
          size_d = std::ceil(
              static_cast<double>(end.to<double>() - start.to<double>()) /
              step.to<double>());
        }

        TORCH_CHECK(
            size_d >= 0 &&
                size_d <=
                    static_cast<double>(std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");
        TORCH_CHECK(
            size_d <= std::numeric_limits<int32_t>::max(),
            "The element number of output should be less than 2^31 "
            "in cnnl_arange_out.");

        int64_t size = static_cast<int64_t>(size_d);
        int64_t numel = out.numel();
        if (numel != size) {
          if (numel > 0) {
            TORCH_WARN(
                "The number of elements in the out tensor of shape ",
                out.sizes(),
                " is ",
                numel,
                " which does not match the computed number of elements ",
                size,
                ". Note that this may occur as a result of rounding error. "
                "The out tensor will be resized to a tensor of shape (",
                size,
                ",).");
          }
          out.resize_(size);
        }

        at::Tensor out_contiguous = cnnl_contiguous(out);

        using cpp_scalar_t = torch_mlu::MLUAccumulateType_t<scalar_t>;
        cpp_scalar_t start_internal = static_cast<cpp_scalar_t>(xstart);
        cpp_scalar_t step_internal = static_cast<cpp_scalar_t>(xstep);
        switch (out_contiguous.scalar_type()) {
          case at::ScalarType::Half:
          case at::ScalarType::BFloat16:
          case at::ScalarType::Float:
          case at::ScalarType::Double:
          case at::ScalarType::Int:
          case at::ScalarType::Long:
            cnnl_arange_internal(
                out_contiguous, &start_internal, &step_internal);
            break;
          default:
            // All floating type is supported, so just handle integral type.
            auto output_internal = at::empty(
                out_contiguous.sizes(),
                out_contiguous.options().dtype(at::kInt));
            cnnl_arange_internal(
                output_internal, &start_internal, &step_internal);
            cnnl_cast_internal(output_internal, out_contiguous);
        }

        if (is_copy_necessary(out, out_contiguous)) {
          out.copy_(out_contiguous);
        }
      });
  return out;
}

} // namespace ops
} // namespace torch_mlu
