#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

using at::native::exp_stub;

void exp_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_exp_internal(output, iter.input(0));
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(exp_stub, &exp_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
