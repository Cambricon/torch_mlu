#include "ATen/ExpandUtils.h"
#include "ATen/native/PointwiseOps.h"

#include "aten/DispatchStub.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/TensorIteratorBridge.h"

using at::native::addcmul_stub;
using at::native::addcmul_stub_DECLARE_DISPATCH_type;

namespace torch_mlu {
namespace ops {

static void addcmul_mlu_kernel(
    at::TensorIteratorBase& iter,
    const at::Scalar& value) {
  if (iter.numel() == 0)
    return;
  auto output = iter.output(0);
  auto self = iter.input(0);
  auto other1 = iter.input(1);
  auto other2 = iter.input(2);
  cnnl_addcmul_internal(output, self, other1, other2, value);
}

REGISTER_PRIVATEUSE1_DISPATCH(addcmul_stub, &addcmul_mlu_kernel);

} // namespace ops
} // namespace torch_mlu
