#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_log_internal(
    at::Tensor& output,
    const at::Tensor& input,
    cnnlLogBase_t base) {
  if (input.numel() == 0) {
    return output;
  }
  // integral type input will be converted to floating before enter kernel
  TORCH_MLU_CHECK(
      at::isFloatingType(input.scalar_type()),
      "log only support floating/integral type");
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  auto desc_input = getTensorDesc(input_impl);
  auto desc_output = getTensorDesc(output_impl);
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlLog(
      handle,
      base,
      desc_input.get(),
      input_ptr,
      desc_output.get(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
