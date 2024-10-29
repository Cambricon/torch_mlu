#include "ATen/native/LinearAlgebraUtils.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/utils/cnnl_util.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include <torch/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor&, at::Tensor&> cnnl_linalg_slogdet_out(
    const at::Tensor& A,
    at::Tensor& sign,
    at::Tensor& logabsdet) {
  TORCH_CHECK(
      A.dim() >= 2, "torch.slogdet: input must have at least 2 dimensions.");
  bool req_grad = torch::autograd::compute_requires_grad(A);
  TORCH_CHECK(
      !req_grad,
      "_linalg_slogdet(): functions with out=... arguments don't support "
      "automatic differentiation, but one of the arguments requires grad.");

  TORCH_CHECK(
      A.dtype() == sign.dtype(),
      "Expected out tensor to have dtype ",
      A.dtype(),
      ", but got ",
      sign.dtype(),
      " instead");
  TORCH_CHECK(
      A.dtype() == logabsdet.dtype(),
      "Expected out tensor to have dtype ",
      A.dtype(),
      ", but got ",
      logabsdet.dtype(),
      " instead");
  at::native::squareCheckInputs(A, "torch.slogdet");
  auto dtype = A.scalar_type();
  TORCH_MLU_CHECK(
      at::isFloatingType(dtype),
      "linalg.slogdet",
      ": Expected a floating point as input. Got ",
      dtype);

  TORCH_MLU_CHECK(
      dtype == at::kFloat || dtype == at::kDouble,
      "linalg.slogdet: Low precision dtypes not supported. Got ",
      dtype);

  at::IntArrayRef out_sizes(A.sizes().data(), A.dim() - 2);
  at::native::resize_output(sign, out_sizes);
  at::native::resize_output(logabsdet, out_sizes);
  auto memory_format = c10::MemoryFormat::Contiguous;
  auto input_A = cnnl_contiguous(A, memory_format);

  // Zero element case
  if (input_A.numel() == 0) {
    if (A.dim() > 2 && sign.numel() == 0) {
      // empty case do nothing.
    } else {
      // sign size = [0, 0] and (A.dim() > 2 && sign.numel() != 0) case
      sign.fill_(1);
      logabsdet.fill_(0);
    }
  } else {
    cnnlDetMode_t mode = CNNL_DET_MODE_SLOGDET;
    std::optional<at::Tensor> sign_opt = sign;
    cnnl_det_internal(logabsdet, input_A, sign_opt, mode);
  }

  return std::forward_as_tuple(sign, logabsdet);
}

// This code was copied from pytorch 1.9
// TODO(zhiguangda): [JIRA-9646] Delete this backward function when cnnl
// supports LU factor operator, and use backward function of origin
Tensor cnnl_linalg_slogdet_backward(
    const Tensor& grad_logabsdet,
    const Tensor& self,
    const Tensor& signdet,
    const Tensor& logabsdet) {
  auto singular_case_backward = [&](const Tensor& grad_logabsdet,
                                    const Tensor& self) -> Tensor {
    Tensor u, sigma, vh;
    std::tie(u, sigma, vh) = at::linalg_svd(self, false);
    Tensor v = vh.conj().transpose(-2, -1);
    // sigma has all non-negative entries (also with at least one zero entry)
    // so logabsdet = \sum log(abs(sigma))
    // but det = 0, so backward logabsdet = \sum log(sigma)
    auto gsigma = grad_logabsdet.unsqueeze(-1).div(sigma);
    return svd_backward({{}, gsigma, {}}, self, true, true, u, sigma, v);
  };

  auto nonsingular_case_backward = [&](const Tensor& grad_logabsdet,
                                       const Tensor& self) -> Tensor {
    // TODO: replace self.inverse with linalg_inverse
    return unsqueeze_multiple(grad_logabsdet, {-1, -2}, self.dim()) *
        self.inverse().conj().transpose(-2, -1);
  };

  if (self.dim() == 2) {
    bool is_singular = self.is_complex() ? signdet.abs().item<double>() == 0
                                         : signdet.item<double>() == 0;
    if (is_singular) {
      return singular_case_backward(grad_logabsdet, self);
    } else {
      return nonsingular_case_backward(grad_logabsdet, self);
    }
  } else {
    auto nonzero_signdet_indices = at::native::toListOfOptionalTensors(
        self.is_complex() ? at::where(signdet.abs()) : at::where(signdet));
    c10::optional<Tensor> first_nonzero_signdet_index =
        nonzero_signdet_indices[0];

    if (first_nonzero_signdet_index->size(0) ==
        logabsdet.numel()) { // all log determinants are finite (non-singular)
      return nonsingular_case_backward(grad_logabsdet, self);
    }

    auto zero_signdet_indices =
        at::native::toListOfOptionalTensors(at::where(signdet == 0));
    c10::optional<Tensor> first_zero_signdet_index = zero_signdet_indices[0];

    if (first_zero_signdet_index->size(0) ==
        logabsdet.numel()) { // all log determinants are -inf (singular)
      return singular_case_backward(grad_logabsdet, self);
    }

    Tensor grad_slogdet = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    // invertible case
    grad_slogdet.index_put_(
        /*indices=*/nonzero_signdet_indices,
        // NOLINTNEXTLINE(bugprone-argument-comment)
        /*value=*/
        nonsingular_case_backward(
            grad_logabsdet.index(nonzero_signdet_indices),
            self.index(nonzero_signdet_indices)));

    // non-invertible case, uses SVD
    grad_slogdet.index_put_(
        /*indices=*/zero_signdet_indices,
        // NOLINTNEXTLINE(bugprone-argument-comment)
        /*value=*/
        singular_case_backward(
            grad_logabsdet.index(zero_signdet_indices),
            self.index(zero_signdet_indices)));

    return grad_slogdet;
  }
}

class SlogdetFunction : public torch::autograd::Function<SlogdetFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& A) {
    at::AutoDispatchBelowADInplaceOrView g;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("aten::linalg_slogdet", "")
                         .typed<decltype(cnnl_linalg_slogdet)>();
    auto result = op.call(A);

    auto result0 = std::get<0>(result);
    auto result1 = std::get<1>(result);
    ctx->save_for_backward({A, result0, result1});
    return {result0, result1};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    // Complex type is not supported yet, only consider logabsdet grads
    if (grad_output[1].defined()) {
      auto grad_logabsdet = grad_output[1];
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torch_mlu::linalg_slogdet_backward", "")
              .typed<decltype(cnnl_linalg_slogdet_backward)>();
      auto result = op.call(grad_logabsdet, saved[0], saved[1], saved[2]);
      return {result};
    } else {
      return {};
    }
  }
};

std::tuple<at::Tensor, at::Tensor> cnnl_linalg_slogdet(const Tensor& A) {
  TORCH_CHECK(
      A.dim() >= 2, "torch.slogdet: input must have at least 2 dimensions.");
  auto shape = A.sizes();
  auto ndim = shape.size();

  auto shape_outputs = shape.slice(0, ndim - 2);
  auto sign = at::empty(shape_outputs, A.options());
  auto logabsdet = at::empty(shape_outputs, A.options());

  auto result = cnnl_linalg_slogdet_out(A, sign, logabsdet);
  return result;
}

std::tuple<at::Tensor, at::Tensor> cnnl_linalg_slogdet_autograd(
    const Tensor& A) {
  auto result = SlogdetFunction::apply(A);
  return std::make_tuple(result[0], result[1]);
}

} // namespace ops
} // namespace torch_mlu
