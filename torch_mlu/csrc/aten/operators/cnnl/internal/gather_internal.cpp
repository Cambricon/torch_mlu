#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_gather_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  auto input_impl = getMluTensorImpl(self);
  auto input_desc = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto index_impl = getMluTensorImpl(index);
  auto index_desc = getTensorDesc(index_impl, CNNL_LAYOUT_ARRAY);
  auto indices_ptr = mlu_data_ptr(index_impl);

  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // set descriptor config
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetGatherWorkspaceSize(
      handle,
      input_desc.get(),
      index_desc.get(),
      output_desc.get(),
      dim,
      &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
  TORCH_CNNL_CHECK(cnnlGather_v2(
      handle,
      dim,
      input_desc.get(),
      input_ptr,
      index_desc.get(),
      indices_ptr,
      workspace_ptr.get(),
      space_size,
      output_desc.get(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
