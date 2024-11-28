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
  auto desc_input = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto desc_output = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlLog1p_v2(
      handle, desc_input.get(), input_ptr, desc_output.get(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
