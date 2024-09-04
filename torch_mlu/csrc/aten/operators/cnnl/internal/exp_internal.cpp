#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_exp_internal(at::Tensor& output, const at::Tensor& input) {
  TORCH_CHECK(
      at::isFloatingType(input.scalar_type()),
      "exp only support floating type");
  if (input.numel() == 0) {
    return output;
  }
  auto input_impl = getMluTensorImpl(input);
  auto descInput = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto descOutput = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // set descriptor config
  const cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  TORCH_CNNL_CHECK(cnnlExp_v2(
      handle,
      prefer,
      descInput.get(),
      input_ptr,
      descOutput.get(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
