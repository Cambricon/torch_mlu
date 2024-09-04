#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_expm1_internal(at::Tensor& output, const at::Tensor& input) {
  TORCH_CHECK(
      at::isFloatingType(input.scalar_type()),
      "expm1 only support floating type");
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
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  // set descriptor config
  const cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  TORCH_CNNL_CHECK(cnnlExpm1(
      handle,
      prefer,
      descInput.desc(),
      input_ptr,
      descOutput.desc(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
