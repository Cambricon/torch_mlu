#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/TensorIteratorBridge.h"

using at::native::sign_stub;

namespace torch_mlu {
namespace ops {

void sign_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0) {
    return;
  }
  auto output = iter.output(0);
  cnnl_sign_internal(output, iter.input(0));
  iter.cast_outputs();
}

REGISTER_PRIVATEUSE1_DISPATCH(sign_stub, &sign_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
