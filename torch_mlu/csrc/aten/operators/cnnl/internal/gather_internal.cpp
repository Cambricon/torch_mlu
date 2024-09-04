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
  if (self.dim() != index.dim()) {
    input_desc.set_dim(self);
    index_desc.set_dim(index);
    output_desc.set_dim(output);
  } else {
    input_desc.set(self);
    index_desc.set(index);
    output_desc.set(output);
  }
  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto indices_ptr = mlu_data_ptr(index_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  // set descriptor config
  TORCH_CNNL_CHECK(cnnlGather(
      handle,
      dim,
      input_desc.desc(),
      input_ptr,
      index_desc.desc(),
      indices_ptr,
      output_desc.desc(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
