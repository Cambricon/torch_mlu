#include <ATen/native/Resize.h>
#include <ATen/native/SpectralOpsUtils.h>

#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/internal/cnfft_plan_cache.h"

namespace torch_mlu {
namespace ops {

constexpr int64_t cnfft_max_ndim = 1;

// Calculates the normalization constant and applies it in-place to self
// sizes is the sizes of a twosided tensor and dims are all transformed dims
double _fft_normalization_scale(
    int64_t normalization,
    IntArrayRef sizes,
    IntArrayRef dims) {
  auto norm = static_cast<at::native::fft_norm_mode>(normalization);
  if (norm == at::native::fft_norm_mode::none) {
    return 1.0;
  }

  int64_t signal_numel = 1;
  for (auto dim : dims) {
    signal_numel *= sizes[dim];
  }
  const double scale_denom = (norm == at::native::fft_norm_mode::by_root_n)
      ? std::sqrt(signal_numel)
      : static_cast<double>(signal_numel);
  return 1.0 / scale_denom;
}

const Tensor& _fft_apply_normalization(
    const Tensor& self,
    int64_t normalization,
    IntArrayRef sizes,
    IntArrayRef dims) {
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  return (scale == 1.0) ? self : self.mul_(scale);
}

Tensor& _fft_apply_normalization_out(
    Tensor& out,
    const Tensor& self,
    int64_t normalization,
    IntArrayRef sizes,
    IntArrayRef dims) {
  auto scale = _fft_normalization_scale(normalization, sizes, dims);
  return at::mul_out(out, self, c10::scalar_to_tensor(scale));
}

at::Tensor cnnl__fft_r2c(
    const at::Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  TORCH_MLU_CHECK(
      self.numel() > 0, "currently do not support empty Tensor input");
  TORCH_MLU_CHECK(
      self.is_floating_point(), "only support real floating point input");
  // TODO: Remove this when nl support _fft_fill_with_conjugate_symmetry op
  TORCH_MLU_CHECK(onesided, "CNNL FFT currently only support onesided");

  auto input_sizes = self.sizes();
  at::DimVector onesided_sizes(input_sizes.begin(), input_sizes.end());
  auto last_dim = dim.back();
  auto last_dim_halfsize = (input_sizes[last_dim]) / 2 + 1;
  onesided_sizes[last_dim] = last_dim_halfsize;
  IntArrayRef out_sizes = onesided ? onesided_sizes : input_sizes;

  const auto out_options =
      self.options().dtype(c10::toComplexType(self.scalar_type()));
  auto output = at::empty(out_sizes, out_options);

  // Calculate scale_factor
  double scale_factor =
      _fft_normalization_scale(normalization, input_sizes, dim);
  auto scale_factor_float =
      c10::checked_convert<float, double>(scale_factor, "float");

  auto working_tensor = self;
  at::DimVector sorted_dims(dim.begin(), dim.end() - 1);
  // First do the R2C transform on the last dimension
  if (sorted_dims.empty()) {
    cnnl_fft_internal(
        output,
        working_tensor,
        onesided_sizes,
        last_dim,
        /*forward=*/true,
        scale_factor_float);
  } else {
    cnnl_fft_internal(
        output,
        working_tensor,
        onesided_sizes,
        last_dim,
        /*forward=*/true,
        1.0);
  }
  if (dim.size() > 1) {
    working_tensor = at::empty(out_sizes, out_options);
  }

  // Then any remaining C2C transforms
  while (!sorted_dims.empty()) {
    std::swap(output, working_tensor);

    // Resort dimensions every time as _exec_fft re-strides the output
    auto strides = working_tensor.strides();
    std::sort(
        sorted_dims.begin(), sorted_dims.end(), [&](int64_t a, int64_t b) {
          return strides[a] > strides[b];
        });

    const auto max_dims =
        std::min(static_cast<size_t>(cnfft_max_ndim), sorted_dims.size());
    auto last_dims =
        IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);

    if (sorted_dims.size() > 1) {
      cnnl_fft_internal(
          output,
          working_tensor,
          onesided_sizes,
          last_dims,
          /*forward=*/true,
          1.0);
    } else {
      cnnl_fft_internal(
          output,
          working_tensor,
          onesided_sizes,
          last_dims,
          /*forward=*/true,
          scale_factor_float);
    }
    sorted_dims.resize(sorted_dims.size() - max_dims);
  }

  return output;
}

Tensor cnnl__fft_c2r(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t lastdim) {
  TORCH_MLU_CHECK(
      self.numel() > 0, "currently do not support empty Tensor input");
  TORCH_MLU_CHECK(self.is_complex(), "only support complex dtype input");
  auto in_sizes = self.sizes();
  at::DimVector out_sizes(in_sizes.begin(), in_sizes.end());
  out_sizes[dim.back()] = lastdim;

  // First complete any C2C transforms
  Tensor tmp = self;
  if (dim.size() > 1) {
    tmp = cnnl__fft_c2c(
        self,
        dim.slice(0, dim.size() - 1),
        static_cast<int64_t>(at::native::fft_norm_mode::none),
        /*forward=*/false);
  }

  // Finally, do a 1D C2R transform
  auto output = at::empty(
      out_sizes,
      self.options().dtype(c10::toRealValueType(self.scalar_type())));
  cnnl_fft_internal(output, tmp, out_sizes, dim.back(), /*forward=*/false, 1.0);
  return _fft_apply_normalization(output, normalization, out_sizes, dim);
}

at::Tensor& cnnl__fft_c2r_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    at::Tensor& out) {
  auto result = cnnl__fft_c2r(
      self,
      dim,
      static_cast<int64_t>(at::native::fft_norm_mode::none),
      last_dim_size);
  return _fft_apply_normalization_out(
      out, result, normalization, result.sizes(), dim);
}

// n-dimensional complex to complex FFT/IFFT
Tensor cnnl__fft_c2c(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_MLU_CHECK(
      self.numel() > 0, "currently do not support empty Tensor input");
  TORCH_MLU_CHECK(self.is_complex(), "only support complex dtype input");
  if (dim.empty()) {
    return self.clone();
  }

  auto out_sizes = self.sizes();
  auto output = at::empty(out_sizes, self.options());

  // Calculate scale_factor
  double scale_factor = _fft_normalization_scale(normalization, out_sizes, dim);
  auto scale_factor_ =
      c10::checked_convert<float, double>(scale_factor, "float");

  // Perform any number of C2C transforms
  at::DimVector sorted_dims(dim.begin(), dim.end());
  auto working_tensor = self;
  while (true) {
    // Sort dimensions every time as _exec_fft re-strides the output
    auto strides = working_tensor.strides();
    std::sort(
        sorted_dims.begin(), sorted_dims.end(), [&](int64_t a, int64_t b) {
          return strides[a] > strides[b];
        });

    const auto max_dims =
        std::min(static_cast<size_t>(cnfft_max_ndim), sorted_dims.size());
    auto first_dims =
        IntArrayRef(sorted_dims).slice(sorted_dims.size() - max_dims, max_dims);

    if (sorted_dims.size() > 1) {
      cnnl_fft_internal(
          output, working_tensor, out_sizes, first_dims, forward, 1.0);
    } else {
      cnnl_fft_internal(
          output,
          working_tensor,
          out_sizes,
          first_dims,
          forward,
          scale_factor_);
    }
    sorted_dims.resize(sorted_dims.size() - max_dims);

    if (sorted_dims.empty()) {
      break;
    }

    if (working_tensor.is_same(self)) {
      working_tensor = std::move(output);
      output = at::empty(out_sizes, self.options());
    } else {
      std::swap(output, working_tensor);
    }
  }

  return output;
}

at::Tensor& cnnl__fft_c2c_out(
    const at::Tensor& self,
    at::IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  auto result = cnnl__fft_c2c(
      self,
      dim,
      static_cast<int64_t>(at::native::fft_norm_mode::none),
      forward);
  return _fft_apply_normalization_out(
      out, result, normalization, result.sizes(), dim);
}

} // namespace ops
} // namespace torch_mlu
