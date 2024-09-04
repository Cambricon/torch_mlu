#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "ATen/WrapDimUtilsMulti.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_flip(const at::Tensor& self, at::IntArrayRef dims) {
  const int64_t total_dims = self.dim();
  // It wraps the dims and checks that there are no repeated dims
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);

  auto out_tensor = at::empty_like(self, self.suggest_memory_format());

  // Count dimensions in which we need to do work
  int n = 0;
  for (const auto i : c10::irange(total_dims)) {
    if (flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      n++;
    }
  }

  // Nothing to do, we return fast
  if (n == 0 || self.numel() <= 1) {
    out_tensor.copy_(self);
    return out_tensor;
  }

  auto self_contiguous = cnnl_contiguous(self, self.suggest_memory_format());
  cnnl_flip_internal(self_contiguous, out_tensor, dims);
  return out_tensor;
}

} // namespace ops
} // namespace torch_mlu
