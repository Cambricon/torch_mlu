#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_erfinv_internal(at::Tensor& output, const at::Tensor& self) {
  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto desc_self = getTensorDesc(self_impl, CNNL_LAYOUT_ARRAY);
  auto self_ptr = mlu_data_ptr(self_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // compute ops
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  TORCH_CNNL_CHECK(cnnlErfinv(
      handle,
      prefer,
      desc_self.get(),
      self_ptr,
      output_desc.get(),
      output_ptr));

  return output;
}
} // namespace ops
} // namespace torch_mlu
