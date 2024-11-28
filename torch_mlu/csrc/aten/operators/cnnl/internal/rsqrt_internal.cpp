#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_rsqrt_internal(at::Tensor& output, const at::Tensor& input) {
  if (input.numel() == 0) {
    return output;
  }
  // Integral type input will be converted to float before enter kernel
  TORCH_CHECK(
      at::isFloatingType(input.scalar_type()),
      "rsqrt only support floating/integral type");

  auto input_impl = getMluTensorImpl(input);
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  TORCH_CNNL_CHECK(cnnlRsqrt(
      handle, input_desc.get(), input_ptr, output_desc.get(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
