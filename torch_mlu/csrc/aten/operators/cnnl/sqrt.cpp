#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

using at::native::sqrt_stub;
using at::native::sqrt_stub_DECLARE_DISPATCH_type;

namespace torch_mlu {
namespace ops {

void sqrt_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_sqrt_internal(output, iter.input(0));
}

REGISTER_PRIVATEUSE1_DISPATCH(sqrt_stub, &sqrt_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
