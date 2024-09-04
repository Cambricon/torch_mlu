#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_logaddexp_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& other,
    LogaddexpType base) {
  auto input_impl = getMluTensorImpl(input);
  auto desc_input = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto other_impl = getMluTensorImpl(other);
  auto desc_other = getTensorDesc(other_impl);
  auto other_ptr = mlu_data_ptr(other_impl);

  auto output_impl = getMluTensorImpl(output);
  auto desc_output = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  size_t space_size = 0;
  // logaddexp
  if (base == LogaddexpType::LOGADDEXP_BASE_E) {
    TORCH_CNNL_CHECK(cnnlGetLogAddExpWorkspaceSize(
        handle,
        desc_input.get(),
        desc_other.get(),
        desc_output.get(),
        &space_size));
    auto workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
    TORCH_CNNL_CHECK(cnnlLogAddExp(
        handle,
        desc_input.get(),
        input_ptr,
        desc_other.get(),
        other_ptr,
        workspace_ptr.get(),
        space_size,
        desc_output.get(),
        output_ptr));
  }
  // logaddexp2
  else if (base == LogaddexpType::LOGADDEXP_BASE_2) {
    TORCH_CNNL_CHECK(cnnlGetLogAddExp2WorkspaceSize(
        handle,
        desc_input.get(),
        desc_other.get(),
        desc_output.get(),
        &space_size));
    auto workspace_ptr =
        torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
    TORCH_CNNL_CHECK(cnnlLogAddExp2(
        handle,
        desc_input.get(),
        input_ptr,
        desc_other.get(),
        other_ptr,
        workspace_ptr.get(),
        space_size,
        desc_output.get(),
        output_ptr));
  } else
    TORCH_CHECK(false, "Illegal logaddexp_base was given");
}

} // namespace ops
} // namespace torch_mlu
