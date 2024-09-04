#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/utils/cnnl_util.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "ATen/native/LinearAlgebraUtils.h"
#include "aten/generated/MLUFunctions.h"
namespace torch_mlu {
namespace ops {

void resize_out(
    const at::Tensor& out,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  TORCH_CHECK(
      options.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      options.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      options.device() == out.device(),
      "Expected out tensor to have device ",
      options.device(),
      ", but got ",
      out.device(),
      " instead");
  const bool resized = at::native::resize_output(out, sizes);
  // Only restride if a resize occurred; otherwise we ignore the (advisory)
  // strides from the meta function and directly use the output tensor's
  // preexisting strides
  if (resized) {
    if (!strides.empty()) {
      TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
      // TODO: avoid the redispatch here
      out.as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
      out.unsafeGetTensorImpl()->empty_tensor_restride(
          *options.memory_format_opt());
    }
  }
}

Tensor create_out(
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  return torch_mlu::mlu::empty_strided(sizes, strides, options);
}

std::optional<at::Tensor> maybe_create_proxy(
    const at::Tensor& out,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options) {
  if (out.strides() != strides) {
    return torch_mlu::mlu::empty_strided(sizes, strides, options);
  }
  return c10::nullopt;
}

// The following 2 functions actually do not follow the native dispatch logic,
// since the native dispatch path is inverse -> linalg_inv -> linalg_inv_ex ->
// linalg_inv_ex_out -> linalg_solve_ex_out -> linalg_lu_solve_out
// (linalg_lu_factor_ex_out) -> cuda kernels. Currently we do not have CNNL
// kernels correspond to linalg_lu_solve_out and linalg_lu_factor_ex_out. In
// addition, we cannot register linalg_inv and inverse directly since they are
// inplicitautograd. Therefore we temporarily registered the following ops and
// ignored the structred delelegate. The dispatch path of inverse has now became
// inverse -> linalg_inv -> linalg_inv_ex -> linalg_inv_ex_out -> cnnlInverse.
// However, the following 2 functions are incomplete since they can only
// calculate the inverted matrices but their infos (we used zeros(1) as insted).
// TODO(PYTORCH-9454)
// Re-adapt the following 2 ops when CNNL linalg_lu_factor and linalg_lu_solve
// kernels are ready.

// In addition, due to the current limitation of torch_mlu codegen, native
// TORCH_META_FUNC cannot be reused by far. When the native dispatch path (CPU/
// CUDA) of a structured kernel points towards one same function (i.e. both CPU
// and CUDA use the same TORCH_IMPL_FUNC), codegen module in torch_mlu will not
// declare a mlu TORCH_IMPL_FUNC for MLU, it makes MLU tensors to use the native
// TORCH_IMPL_FUNC instead.
// TODO(PYTORCH-10057)
// Use native TORCH_META_FUNC when codegen module supports overridable
// TORCH_IMPL_FUNC for structured kernels with the same CPU and CUDA dispatch
// pathes.

std::tuple<at::Tensor, at::Tensor> cnnl_linalg_inv_ex(
    const Tensor& A,
    bool check_errors) {
  at::native::squareCheckInputs(A, "linalg.inv");
  at::native::checkFloatingOrComplex(
      A, "linalg.inv", /*allow_low_precision_dtypes*/ false);
  auto shape = A.sizes();
  auto result_strides =
      at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  auto result = create_out(shape, result_strides, A.options());
  auto info = at::empty(
      {at::native::batchCount(A)}, A.options().dtype(at::ScalarType::Int));
  if (A.numel() == 0) {
    info.zero_();
  }
  auto memory_format = c10::MemoryFormat::Contiguous;
  auto A_contiguous = cnnl_contiguous(A, memory_format);
  cnnl_inverse_internal(result.transpose(-1, -2), A_contiguous, info);
  return std::make_tuple(result, info);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_linalg_inv_ex_out(
    const Tensor& A,
    bool check_errors,
    Tensor& result,
    Tensor& info) {
  at::native::squareCheckInputs(A, "linalg.inv");
  at::native::checkFloatingOrComplex(
      A, "linalg.inv", /*allow_low_precision_dtypes*/ false);
  auto info_ = info;
  if (info.numel() == 0) {
    info_ = at::zeros(
        {at::native::batchCount(A)}, A.options().dtype(at::ScalarType::Int));
  }
  auto memory_format = c10::MemoryFormat::Contiguous;
  auto shape = A.sizes();
  auto A_contiguous = cnnl_contiguous(A, memory_format);
  auto result_strides =
      at::native::batched_matrix_contiguous_strides(shape, /*f-contig*=*/true);
  resize_out(result, shape, result_strides, A.options());
  auto maybe_proxy =
      maybe_create_proxy(result, shape, result_strides, A.options());
  if (C10_UNLIKELY(maybe_proxy.has_value())) {
    auto result_ = std::move(maybe_proxy).value();
    cnnl_inverse_internal(result_.transpose(-1, -2), A_contiguous, info_);
    result.copy_(result_);
    return std::forward_as_tuple(result, info);
  }
  cnnl_inverse_internal(result.transpose(-1, -2), A_contiguous, info_);
  return std::forward_as_tuple(result, info);
}

} // namespace ops
} // namespace torch_mlu
