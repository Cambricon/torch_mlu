#include "ATen/native/BinaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

using at::native::logaddexp2_stub;
using at::native::logaddexp2_stub_DECLARE_DISPATCH_type;
using at::native::logaddexp_stub;
using at::native::logaddexp_stub_DECLARE_DISPATCH_type;

void logaddexp_kernel_mlu(at::TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "logaddexp_mlu",
      [&]() {
        auto self = iter.input(0);
        auto other = iter.input(1);
        auto output = iter.output(0);
        cnnl_logaddexp_internal(
            output, self, other, LogaddexpType::LOGADDEXP_BASE_E);
      });
}

void logaddexp2_kernel_mlu(at::TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.dtype(),
      "logaddexp2_mlu",
      [&]() {
        auto self = iter.input(0);
        auto other = iter.input(1);
        auto output = iter.output(0);
        cnnl_logaddexp_internal(
            output, self, other, LogaddexpType::LOGADDEXP_BASE_2);
      });
}

REGISTER_PRIVATEUSE1_DISPATCH(logaddexp_stub, &logaddexp_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(logaddexp2_stub, &logaddexp2_kernel_mlu);
} // namespace ops
} // namespace torch_mlu
