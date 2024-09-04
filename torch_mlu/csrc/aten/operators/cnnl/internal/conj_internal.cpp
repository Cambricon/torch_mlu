#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/binaryops_util.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_conj_internal(at::Tensor& output, const at::Tensor& input) {
  // get tensor impl
  auto input_impl = getMluTensorImpl(input);
  auto desc_input = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // call cnnl conj api
  TORCH_CNNL_CHECK(cnnlConj(
      handle, desc_input.get(), input_ptr, desc_output.get(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
