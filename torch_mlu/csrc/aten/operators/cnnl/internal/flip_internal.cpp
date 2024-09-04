#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"
#include <ATen/WrapDimUtilsMulti.h> // NOLINT
#include <algorithm>

namespace torch_mlu {
namespace ops {

void cnnl_flip_internal(
    const at::Tensor& self,
    at::Tensor& output,
    at::IntArrayRef dims) {
  int self_dim = self.dim();
  int len = dims.size();
  c10::SmallVector<int, 8> dim(len, 0);

  for (int i = 0; i < len; ++i) {
    dim[i] = ::at::maybe_wrap_dim(dims[i], self_dim);
  }

  std::transform(dim.begin(), dim.end(), dim.begin(), [&self](const int64_t d) {
    return modify_dim_based_on_layout(d, self.suggest_memory_format());
  });

  // input
  auto input_impl = getMluTensorImpl(self);
  auto input_desc = getTensorDesc(input_impl);
  auto input_ptr = mlu_data_ptr(input_impl);

  // output
  auto output_impl = getMluTensorImpl(output);
  auto output_desc = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // get current handle
  auto handle = getCurrentHandle();

  AT_DISPATCH_MLU_TENSOR_SCLAER_TYPES(self.scalar_type(), "flip", [&] {
    TORCH_CNNL_CHECK(cnnlFlip(
        handle,
        dim.data(),
        len,
        input_desc.get(),
        input_ptr,
        output_desc.get(),
        output_ptr));
  });
}

} // namespace ops
} // namespace torch_mlu
