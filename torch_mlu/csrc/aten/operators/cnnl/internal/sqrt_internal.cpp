#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_sqrt_internal(at::Tensor& output, const at::Tensor& input) {
  if (input.numel() == 0) {
    return output;
  }
  // Integral type input will be converted to float before enter kernel
  TORCH_MLU_CHECK(
      at::isFloatingType(input.scalar_type()),
      "sqrt only support floating/integral type");
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  auto desc_input = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto desc_output = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  cnnlComputationPreference_t mode = CNNL_COMPUTATION_HIGH_PRECISION;

  TORCH_CNNL_CHECK(cnnlSqrt_v2(
      handle,
      mode,
      desc_input.get(),
      input_ptr,
      desc_output.get(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
