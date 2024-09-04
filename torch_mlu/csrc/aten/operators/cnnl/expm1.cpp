#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

using at::native::expm1_stub;

void expm1_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_expm1_internal(output, iter.input(0));
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(expm1_stub, &expm1_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
