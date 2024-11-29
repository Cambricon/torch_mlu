#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_exp_internal(at::Tensor& output, const at::Tensor& input) {
  TORCH_MLU_CHECK(
      at::isFloatingType(input.scalar_type()),
      "exp only support floating type");
  if (input.numel() == 0) {
    return output;
  }
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  descInput.set(input, CNNL_LAYOUT_ARRAY);
  descOutput.set(output, CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlExp(
      handle, descInput.desc(), input_ptr, descOutput.desc(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
