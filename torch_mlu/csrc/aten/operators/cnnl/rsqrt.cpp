#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

using at::native::rsqrt_stub;
using at::native::rsqrt_stub_DECLARE_DISPATCH_type;

namespace torch_mlu {
namespace ops {

void rsqrt_mlu_kernel(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  cnnl_rsqrt_internal(output, iter.input(0));
}

REGISTER_PRIVATEUSE1_DISPATCH(rsqrt_stub, &rsqrt_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
