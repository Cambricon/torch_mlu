#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_floor_internal(at::Tensor& output, const at::Tensor& input) {
  if (input.numel() == 0) {
    return output;
  }
  TORCH_MLU_CHECK(
      at::isFloatingType(input.scalar_type()),
      "floor input only support floating type");
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlFloor(
      handle, input_desc.get(), input_ptr, output_desc.get(), output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
