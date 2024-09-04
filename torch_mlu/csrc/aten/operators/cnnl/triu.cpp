#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

TORCH_IMPL_FUNC(triu_out_mlu)
(const Tensor& self, int64_t k, const Tensor& output) {
  if (self.numel() != 0) {
    auto self_contiguous = cast_long_to_int_if_needed(cnnl_contiguous(self));
    at::Tensor output_contiguous;
    output_contiguous = create_int_tensor_if_needed(cnnl_contiguous(output));
    cnnl_tri_internal(output_contiguous, self_contiguous, k, true);
    if (!output_contiguous.is_same(output)) {
      output.copy_(output_contiguous);
    }
  }
}

} // namespace ops
} // namespace torch_mlu
