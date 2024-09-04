#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"
#include "aten/TensorIteratorBridge.h"

using at::native::sgn_stub;
using at::native::sgn_stub_DECLARE_DISPATCH_type;

namespace torch_mlu {
namespace ops {

void sgn_mlu_kernel(at::TensorIteratorBase& iter) {
  if (iter.numel() == 0) {
    return;
  }
  auto output = iter.output(0);
  cnnl_sign_internal(output, iter.input(0));
}

REGISTER_PRIVATEUSE1_DISPATCH(sgn_stub, &sgn_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
