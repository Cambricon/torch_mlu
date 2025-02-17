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

  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetInverseWorkspaceSize(
      handle,
      desc_input.desc(),
      desc_output.desc(),
      desc_infos.desc(),
      &workspace_size));
  auto temp_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlInverse_v2(
      handle,
      desc_input.desc(),
      input_ptr,
      true,
      temp_ptr.get(),
      workspace_size,
      desc_output.desc(),
      output_ptr,
      desc_infos.desc(),
      infos_ptr));
}

} // namespace ops
} // namespace torch_mlu
