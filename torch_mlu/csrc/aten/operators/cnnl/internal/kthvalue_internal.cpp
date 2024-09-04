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
  auto self_desc = getTensorDesc(self_impl);
  auto self_ptr = mlu_data_ptr(self_impl);

  auto values_impl = getMluTensorImpl(values);
  auto values_desc = getTensorDesc(values_impl);
  auto values_ptr = mlu_data_ptr(values_impl);

  auto indices_impl = getMluTensorImpl(indices);
  auto indices_desc = getTensorDesc(indices_impl, CNNL_DTYPE_INT32);
  auto indices_ptr = mlu_data_ptr(indices_impl);

  // get current handle
  auto handle = getCurrentHandle();

  // calculate
  TORCH_CNNL_CHECK(cnnlKthValue(
      handle,
      self_desc.get(),
      self_ptr,
      k,
      dim,
      values_desc.get(),
      values_ptr,
      indices_desc.get(),
      indices_ptr));
}

} // namespace ops
} // namespace torch_mlu
