#include "ATen/native/UnaryOps.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/DispatchStub.h"

namespace torch_mlu {
namespace ops {

using at::native::erfinv_stub;
using at::native::erfinv_stub_DECLARE_DISPATCH_type;

void erfinv_kernel_mlu(at::TensorIteratorBase& iter) {
  auto output = iter.output(0);
  auto self = iter.input(0);
  cnnl_erfinv_internal(output, self);
}

REGISTER_PRIVATEUSE1_DISPATCH(erfinv_stub, &erfinv_kernel_mlu);

} // namespace ops
} // namespace torch_mlu
