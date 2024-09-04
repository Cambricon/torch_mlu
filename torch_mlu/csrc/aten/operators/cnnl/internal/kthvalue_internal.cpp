#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

void cnnl_kthvalue_internal(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim) {
  TORCH_CHECK(self.dim() >= 0, "dimension not support");
  auto memory_format = self.suggest_memory_format();
  dim = modify_dim_based_on_layout(dim, memory_format);

  auto self_impl = getMluTensorImpl(self);
  auto values_impl = getMluTensorImpl(values);
  auto indices_impl = getMluTensorImpl(indices);

  // get cnnl sizes and strides
  auto self_sizes_strides = get_tensor_size_stride(self, memory_format);
  auto vaules_sizes_strides = get_tensor_size_stride(values, memory_format);
  auto indices_sizes_strides = get_tensor_size_stride(indices, memory_format);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTensorDescriptor self_desc;
  CnnlTensorDescriptor values_desc;
  CnnlTensorDescriptor indices_desc;

  // get cnnl descriptor
  self_desc.set(
      self,
      std::get<0>(self_sizes_strides),
      std::get<1>(self_sizes_strides),
      CNNL_LAYOUT_ARRAY);
  values_desc.set(
      values,
      std::get<0>(vaules_sizes_strides),
      std::get<1>(vaules_sizes_strides),
      CNNL_LAYOUT_ARRAY);
  indices_desc.set(
      indices,
      std::get<0>(indices_sizes_strides),
      std::get<1>(indices_sizes_strides),
      CNNL_LAYOUT_ARRAY,
      CNNL_DTYPE_INT32);

  // malloc mlu memory
  auto self_ptr = self_impl->mlu_data_ptr();
  auto values_ptr = values_impl->mlu_data_ptr();
  auto indices_ptr = indices_impl->mlu_data_ptr();

  // calculate
  TORCH_CNNL_CHECK(cnnlKthValue(
      handle,
      self_desc.desc(),
      self_ptr,
      k,
      dim,
      values_desc.desc(),
      values_ptr,
      indices_desc.desc(),
      indices_ptr));
}

} // namespace ops
} // namespace torch_mlu
