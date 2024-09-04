#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

using at::native::log10_stub;
using at::native::log1p_stub;
using at::native::log2_stub;
using at::native::log_stub;

void log_kernel_mlu(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_log_internal(output, iter.input(0), CNNL_LOG_E);
  iter.cast_outputs();
}

void log10_kernel_mlu(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_log_internal(output, iter.input(0), CNNL_LOG_10);
  iter.cast_outputs();
}

void log2_kernel_mlu(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_log_internal(output, iter.input(0), CNNL_LOG_2);
  iter.cast_outputs();
}

void log1p_kernel_mlu(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_log1p_internal(output, iter.input(0));
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(log_stub, &log_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(log10_stub, &log10_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(log1p_stub, &log1p_kernel_mlu);
REGISTER_PRIVATEUSE1_DISPATCH(log2_stub, &log2_kernel_mlu);

} // namespace ops
} // namespace torch_mlu
