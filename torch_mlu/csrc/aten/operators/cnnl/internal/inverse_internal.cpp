#include "ATen/native/LinearAlgebraUtils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_inverse_internal(
    const at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& infos) {
  TORCH_CHECK(
      input.dtype() == at::kFloat || input.dtype() == at::kDouble,
      "inverse operator only supports float and double types");
  CnnlTensorDescriptor desc_input;
  CnnlTensorDescriptor desc_output;
  CnnlTensorDescriptor desc_infos;
  desc_input.set(input);
  desc_output.set(output);
  desc_infos.set(infos);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto infos_impl = getMluTensorImpl(infos);

  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  auto infos_ptr = infos_impl->mlu_data_ptr();

  // set descriptor config
  auto handle = getCurrentHandle();
  TORCH_CNNL_CHECK(cnnlInverse(
      handle,
      desc_input.desc(),
      input_ptr,
      true,
      desc_output.desc(),
      output_ptr,
      desc_infos.desc(),
      infos_ptr));
}

} // namespace ops
} // namespace torch_mlu
