#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_gather_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index) {
  auto input_impl = getMluTensorImpl(self);
  auto index_impl = getMluTensorImpl(index);
  auto output_impl = getMluTensorImpl(output);
  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor input_desc;
  CnnlTensorDescriptor index_desc;
  CnnlTensorDescriptor output_desc;
  input_desc.set(self);
  // cnnlGather_v2 support strided index, just pass the original index stride
  // and shape info to cnnl.
  index_desc.set(index, CNNL_LAYOUT_ARRAY);
  output_desc.set(output);
  // malloc mlu memory
  auto input_ptr = input_impl->mlu_data_ptr();
  auto indices_ptr = index_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetGatherWorkspaceSize(
      handle,
      input_desc.desc(),
      index_desc.desc(),
      output_desc.desc(),
      dim,
      &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);
  TORCH_CNNL_CHECK(cnnlGather_v2(
      handle,
      dim,
      input_desc.desc(),
      input_ptr,
      index_desc.desc(),
      indices_ptr,
      workspace_ptr.get(),
      space_size,
      output_desc.desc(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
