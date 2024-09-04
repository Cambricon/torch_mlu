#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

using at::native::floor_stub;
namespace torch_mlu {
namespace ops {

void floor_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_floor_internal(output, iter.input(0));
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(floor_stub, &floor_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
