#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_erfinv_internal(at::Tensor& output, const at::Tensor& self) {
  // get tensor impl
  auto self_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // create the desc
  CnnlTensorDescriptor desc_self;
  CnnlTensorDescriptor output_desc;
  desc_self.set(self, CNNL_LAYOUT_ARRAY);
  output_desc.set(output, CNNL_LAYOUT_ARRAY);

  // get current handle
  auto handle = getCurrentHandle();

  // get the mlu ptr
  auto self_ptr = mlu_data_ptr(self_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // compute ops
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  TORCH_CNNL_CHECK(cnnlErfinv(
      handle,
      prefer,
      desc_self.desc(),
      self_ptr,
      output_desc.desc(),
      output_ptr));

  return output;
}
} // namespace ops
} // namespace torch_mlu
