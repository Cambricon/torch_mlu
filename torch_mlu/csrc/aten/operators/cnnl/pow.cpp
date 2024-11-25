#include <ATen/native/Pow.h>
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

using at::native::pow_tensor_scalar_stub;
using at::native::pow_tensor_scalar_stub_DECLARE_DISPATCH_type;
using at::native::pow_tensor_tensor_stub;
using at::native::pow_tensor_tensor_stub_DECLARE_DISPATCH_type;

void pow_tensor_tensor_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  // Note: Not support pass one CPU tensor that dim is 0 to CNNL kernel.
  auto self = scalar_to_tensor_with_dtype(iter.input(0), output.scalar_type());
  auto other = scalar_to_tensor_with_dtype(iter.input(1), output.scalar_type());
  cnnl_pow_internal(output, self, other);
}

void pow_tensor_scalar_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& exp_scalar) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_pow_internal(output, self, std::nullopt, exp_scalar);
}

REGISTER_PRIVATEUSE1_DISPATCH(
    pow_tensor_tensor_stub,
    &pow_tensor_tensor_mlu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(
    pow_tensor_scalar_stub,
    &pow_tensor_scalar_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
