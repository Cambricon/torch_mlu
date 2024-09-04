#include "ATen/ExpandUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/binaryops_util.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_conj_internal(at::Tensor& output, const at::Tensor& input) {
  // get tensor impl
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  // create input desc
  CnnlTensorDescriptor desc_input;
  desc_input.set(input, CNNL_LAYOUT_ARRAY);
  // create output desc
  CnnlTensorDescriptor desc_output;
  desc_output.set(output, CNNL_LAYOUT_ARRAY);
  // allocate mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // call cnnl conj api
  TORCH_CNNL_CHECK(cnnlConj(
      handle, desc_input.desc(), input_ptr, desc_output.desc(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
