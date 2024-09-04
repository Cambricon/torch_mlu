#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<Tensor&, Tensor&> cnnl_slogdet_internal(
    const at::Tensor& input,
    at::Tensor& sign,
    at::Tensor& output) {
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  CnnlTensorDescriptor descSign;
  descInput.set(input);
  descOutput.set(output);
  descSign.set(sign);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto sign_impl = getMluTensorImpl(sign);
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto sign_impl_ptr = mlu_data_ptr(sign_impl);
  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlSlogDet(
      handle,
      descInput.desc(),
      input_ptr,
      descOutput.desc(),
      output_ptr,
      descSign.desc(),
      sign_impl_ptr));
  return std::tie(sign, output);
}

} // namespace ops
} // namespace torch_mlu