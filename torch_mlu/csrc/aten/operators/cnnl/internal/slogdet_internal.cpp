#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<Tensor&, Tensor&> cnnl_slogdet_internal(
    const at::Tensor& input,
    at::Tensor& sign,
    at::Tensor& output) {
  auto input_impl = getMluTensorImpl(input);
  auto descInput = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto descOutput = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto sign_impl = getMluTensorImpl(sign);
  auto descSign = getTensorDesc(sign_impl);
  auto sign_impl_ptr = mlu_data_ptr(sign_impl);

  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlSlogDet(
      handle,
      descInput.get(),
      input_ptr,
      descOutput.get(),
      output_ptr,
      descSign.get(),
      sign_impl_ptr));
  return std::tie(sign, output);
}

} // namespace ops
} // namespace torch_mlu