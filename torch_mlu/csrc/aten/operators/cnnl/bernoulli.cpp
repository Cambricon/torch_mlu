#include <limits>
#include "ATen/NativeFunctions.h"
#include "ATen/native/UnaryOps.h"
#include "ATen/native/TensorIterator.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/operators/cnnl/copy.h"
#include "aten/utils/dispatch.h"
#include "aten/DispatchStub.h"
#include "ATen/TensorUtils.h"
#include "aten/TensorIteratorBridge.h"

using at::native::bernoulli_scalar_stub;
using at::native::bernoulli_scalar_stub_DECLARE_DISPATCH_type;
using at::native::bernoulli_tensor_stub;
using at::native::bernoulli_tensor_stub_DECLARE_DISPATCH_type;
using at::native::DispatchStub;

namespace torch_mlu {
namespace ops {

template <typename T>
void bernoulli_kernel_impl(
    at::TensorIteratorBase& iter,
    T& p,
    std::optional<at::Generator> gen_) {
  float min = 0.0, max = 1.0;
  TensorIteratorBridge iter_bridge;
  iter_bridge.to_build(iter, "default");
  auto output = iter.output(0);
  // TODO(sifengyang): bernoulli is implemented by uniform_ + lt_.
  // this problem will be solved in PYTORCH-9363.
  cnnl_uniform_(output, min, max, gen_);
  output.lt_(p);
  iter_bridge.cast_outputs(iter);
}

void bernoulli_tensor_mlu_kernel(
    const at::TensorBase& self,
    const at::TensorBase& p_,
    std::optional<at::Generator> gen_) {
  TORCH_CHECK(
      at::isFloatingType(p_.scalar_type()),
      "expected probabilities tensor to have floating type, got ",
      p_.scalar_type());
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "self tensor of bernoulli_tensor_mlu_kernel to have floating type, got ",
      self.scalar_type());
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  at::Tensor p_mlu(p_);
  bernoulli_kernel_impl(iter, p_mlu, gen_);
}

void bernoulli_scalar_mlu_kernel(
    const at::TensorBase& self,
    double p,
    std::optional<at::Generator> gen) {
  TORCH_CHECK(
      c10::isFloatingType(self.scalar_type()),
      "self dtype of bernoulli_scalar_mlu_kernel not implemented for ",
      self.scalar_type());
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  bernoulli_kernel_impl(iter, p, gen);
}

at::Tensor cnnl_bernoulli(
    const at::Tensor& self,
    std::optional<at::Generator> generator) {
  return at::native::bernoulli(self, generator);
}

at::Tensor& cnnl_bernoulli_(
    at::Tensor& self,
    double p,
    std::optional<at::Generator> generator) {
  return at::native::bernoulli_(self, p, generator);
}

at::Tensor& cnnl_bernoulli_(
    at::Tensor& self,
    const Tensor& p,
    std::optional<at::Generator> generator) {
  return at::native::bernoulli_(self, p, generator);
}

at::Tensor& cnnl_bernoulli_out(
    const at::Tensor& self,
    std::optional<at::Generator> generator,
    at::Tensor& out) {
  return at::native::bernoulli_out(self, generator, out);
}

// Only support inplace operation.
at::Tensor& cnnl_uniform_(
    at::Tensor& self,
    double from,
    double to,
    std::optional<at::Generator> generator) {
  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    cnnl_uniform_(float_tensor, from, to, generator);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "check_uniform_bounds",
        [&] {
          const auto dtype = self.dtype();
          const auto min =
              static_cast<double>(std::numeric_limits<scalar_t>::lowest());
          const auto max =
              static_cast<double>(std::numeric_limits<scalar_t>::max());
          TORCH_CHECK(
              from >= min && from <= max,
              "from",
              " is out of bounds for ",
              dtype);
          TORCH_CHECK(
              to >= min && to <= max, "to", " is out of bounds for ", dtype);
          TORCH_CHECK(
              from <= to,
              "uniform_ expects to return a [from, to) range, but found from=",
              from,
              " > to=",
              to);
          TORCH_CHECK(
              (to - from) <= std::numeric_limits<scalar_t>::max(),
              "uniform_ expects to-from <= std::numeric_limits<",
              toString(self.scalar_type()),
              ">::max(), but found to=",
              to,
              " and from=",
              from,
              " which result in to-from to exceed the limit");
          from = std::min(std::max(from, min), max);
          to = std::max(std::min(to, max), min);
        });
    auto iter = at::TensorIterator::borrowing_nullary_op(self);
    TensorIteratorBridge iter_bridge;
    iter_bridge.to_build(iter, "default");
    auto output = iter.output(0);
    cnnl_uniform_internal(output, generator, from, to);
    iter_bridge.cast_outputs(iter);
  }
  return self;
}

using namespace at::native;
REGISTER_PRIVATEUSE1_DISPATCH(
    bernoulli_tensor_stub,
    &bernoulli_tensor_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    bernoulli_scalar_stub,
    &bernoulli_scalar_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
