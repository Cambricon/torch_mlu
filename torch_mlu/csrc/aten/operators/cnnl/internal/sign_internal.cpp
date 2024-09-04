#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_sign_internal(at::Tensor& output, const at::Tensor& input) {
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  // set descriptor config
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  descInput.set(input, CNNL_LAYOUT_ARRAY);
  descOutput.set(output, CNNL_LAYOUT_ARRAY);
  // get current handle
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlSign(
      handle, descInput.desc(), input_ptr, descOutput.desc(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
