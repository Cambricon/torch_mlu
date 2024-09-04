#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_log1p_internal(at::Tensor& output, const at::Tensor& input) {
  if (input.numel() == 0) {
    return output;
  }
  // Integral type input will be converted to float before enter kernel
  TORCH_CHECK(
      at::isFloatingType(input.scalar_type()),
      "log1p only support floating/integral type");
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_output;
  desc_input.set(input);
  desc_output.set(output);
  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlLog1p(
      handle,
      prefer,
      desc_input.desc(),
      input_ptr,
      desc_output.desc(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
