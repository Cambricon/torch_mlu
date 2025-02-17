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
  auto input_impl = getMluTensorImpl(input);
  auto desc_input = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto output_impl = getMluTensorImpl(output);
  auto desc_output = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  auto infos_impl = getMluTensorImpl(infos);
  auto desc_infos = getTensorDesc(infos_impl);
  auto infos_ptr = mlu_data_ptr(infos_impl);

  auto handle = getCurrentHandle();
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetInverseWorkspaceSize(
      handle,
      desc_input.get(),
      desc_output.get(),
      desc_infos.get(),
      &workspace_size));
  auto temp_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlInverse_v2(
      handle,
      desc_input.get(),
      input_ptr,
      true,
      temp_ptr.get(),
      workspace_size,
      desc_output.get(),
      output_ptr,
      desc_infos.get(),
      infos_ptr));
}

} // namespace ops
} // namespace torch_mlu
