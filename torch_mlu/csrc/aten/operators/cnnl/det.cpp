#include <torch/autograd.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include "aten/utils/cnnl_util.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::set<at::ScalarType> support_types{
    at::ScalarType::Double,
    at::ScalarType::Float,
    at::ScalarType::Half};

bool any_variable_defined(const torch::autograd::variable_list& variables) {
  for (const auto& variable : variables) {
    if (variable.defined()) {
      return true;
    }
  }
  return false;
}

at::Tensor reverse_dim(const at::Tensor& t, int64_t dim) {
  at::Tensor index =
      at::arange(t.size(dim) - 1, -1, -1, t.options().dtype(at::kLong));
  return t.index_select(dim, index);
}

at::Tensor prod_safe_zeros_backward(
    const at::Tensor& grad,
    const at::Tensor& inp,
    int64_t dim) {
  if (inp.size(dim) == 1) {
    return grad;
  }

  auto ones_size = inp.sizes().vec();
  ones_size[dim] = 1;
  at::Tensor ones = at::ones(ones_size, grad.options());
  at::Tensor exclusive_normal_nocp =
      at::cat({ones, inp.narrow(dim, 0, inp.size(dim) - 1)}, dim);
  at::Tensor exclusive_normal = exclusive_normal_nocp.cumprod(dim);

  at::Tensor narrow_reverse =
      reverse_dim(inp.narrow(dim, 1, inp.size(dim) - 1), dim);
  at::Tensor exclusive_reverse_nocp = at::cat({ones, narrow_reverse}, dim);
  at::Tensor exclusive_reverse =
      reverse_dim(exclusive_reverse_nocp.cumprod(dim), dim);

  return grad * (exclusive_normal * exclusive_reverse).conj();
}

at::Tensor prod_backward(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& result) {
  if (input.dim() == 0) {
    return grad;
  }
  at::Tensor zero_idx = (input == 0).nonzero();
  if (zero_idx.numel() == 0) {
    return grad * (result / input).conj();
  } else if (zero_idx.size(0) > 1) {
    return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  } else {
    return prod_safe_zeros_backward(grad, input.contiguous().view(-1), 0)
        .view_as(input);
  }
}

at::Tensor& cnnl_linalg_det_out(const at::Tensor& self, at::Tensor& out) {
  TORCH_CHECK(
      self.dim() >= 2, "torch.det: input must have at least 2 dimensions.");
  bool req_grad = torch::autograd::compute_requires_grad(self);
  TORCH_CHECK(
      !req_grad,
      "_linalg_det(): functions with out=... arguments "
      "don't support automatic differentiation, but one of the arguments requires grad.");
  at::native::squareCheckInputs(self, "linalg.det");
  at::native::checkFloatingOrComplex(self, "linalg.det");
  TORCH_CHECK(
      self.dtype() == out.dtype(),
      "Expected out tensor to have dtype ",
      self.dtype(),
      ", but got ",
      out.dtype(),
      " instead");
  TORCH_CHECK(
      support_types.find(self.scalar_type()) != support_types.end(),
      "cnnl_det not implemented for '",
      self.scalar_type(),
      "'");
  auto memory_format = c10::MemoryFormat::Contiguous;
  auto input_t = cnnl_contiguous(self, memory_format);
  at::IntArrayRef out_sizes(self.sizes().data(), self.dim() - 2);
  at::native::resize_output(out, out_sizes);

  // The native det operator is spliced by multiple operators,
  // and mlu implements the large operator of cnnldet. The empty
  // behavior is processed here to ensure alignment with the native behavior.
  if (self.numel() == 0) {
    if (self.dim() > 2 && out.numel() == 0) {
      // self.sizes() like (0,0,5,5), empty case do nothing.
    } else
      out.fill_(1); // self.sizes() like (5,5,0,0)
  } else {
    cnnlDetMode_t mode = CNNL_DET_MODE_DET;
    cnnl_det_internal(out, input_t, mode);
  }
  return out;
}

at::Tensor cnnl_linalg_det_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& det) {
  auto singular_case_backward = [&](const at::Tensor& grad,
                                    const at::Tensor& self,
                                    const at::Tensor& det) -> at::Tensor {
    at::Tensor u, sigma, vh;
    std::tie(u, sigma, vh) = at::linalg_svd(self, false);
    at::Tensor v = vh.conj().transpose(-2, -1);
    auto gsigma = prod_backward(grad.unsqueeze(-1), sigma, det.unsqueeze(-1));
    return svd_backward({{}, gsigma, {}}, self, true, true, u, sigma, v);
  };

  auto nonsingular_case_backward = [&](const at::Tensor& grad,
                                       const at::Tensor& self,
                                       const at::Tensor& det) -> at::Tensor {
    return unsqueeze_multiple(grad * det, {-1, -2}, self.dim()) *
        self.inverse().transpose(-2, -1);
  };

  if (self.dim() == 2) {
    // TODO(PYTORCH-10291):When the forward result of det is all 0-tensor,
    // the precision error in the reverse direction may be larger.
    if (det.item<double>() == 0) {
      return singular_case_backward(grad, self, det);
    } else {
      return nonsingular_case_backward(grad, self, det);
    }
  } else {
    auto nonzero_det_indices =
        at::native::toListOfOptionalTensors(at::where(det));
    std::optional<at::Tensor> first_nonzero_det_index = nonzero_det_indices[0];

    if (first_nonzero_det_index->size(0) == det.numel()) {
      // all determinants are nonzero (non-singular)
      return nonsingular_case_backward(grad, self, det);
    }

    auto zero_det_indices =
        at::native::toListOfOptionalTensors(at::where(det == 0));
    std::optional<at::Tensor> first_zero_det_index = zero_det_indices[0];

    if (first_zero_det_index->size(0) ==
        det.numel()) { // all determinants are zero (singular)
      return singular_case_backward(grad, self, det);
    }

    at::Tensor grad_det = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

    // invertible case
    grad_det.index_put_(
        /*indices=*/nonzero_det_indices,
        /*value=*/
        nonsingular_case_backward(
            grad.index(nonzero_det_indices),
            self.index(nonzero_det_indices),
            det.index(nonzero_det_indices)));

    // non-invertible case, uses SVD
    grad_det.index_put_(
        /*indices=*/zero_det_indices,
        /*value=*/
        singular_case_backward(
            grad.index(zero_det_indices),
            self.index(zero_det_indices),
            det.index(zero_det_indices)));

    return grad_det;
  }
}

class DetFunction : public torch::autograd::Function<DetFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& self) {
    at::AutoDispatchBelowADInplaceOrView g;
    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow("aten::linalg_det", "")
                         .typed<decltype(cnnl_linalg_det)>();
    auto result = op.call(self);
    ctx->save_for_backward({self, result});
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    if (any_variable_defined(grad_outputs)) {
      static auto op =
          c10::Dispatcher::singleton()
              .findSchemaOrThrow("torch_mlu::linalg_det_backward", "")
              .typed<decltype(cnnl_linalg_det_backward)>();
      auto grad_result = op.call(grad_outputs[0], saved[0], saved[1]);
      return {grad_result};
    } else {
      return {at::Tensor()};
    }
  }
};

at::Tensor cnnl_linalg_det(const at::Tensor& self) {
  // TODO(PYTORCH-9646): The native det operator requires cnnl to support
  // lu_factor.
  auto result = at::empty({0}, self.options());
  cnnl_linalg_det_out(self, result);
  return result;
}

at::Tensor cnnl_linalg_det_autograd(const Tensor& A) {
  auto result = DetFunction::apply(A);
  return result[0];
}

} // namespace ops
} // namespace torch_mlu
