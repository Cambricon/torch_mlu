#include <algorithm>

#include "ATen/NativeFunctions.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/cnnl_util.h"
#include "c10/core/MemoryFormat.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_clone(
    const at::Tensor& src,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  auto memory_format =
      optional_memory_format.value_or(c10::MemoryFormat::Preserve);
  if (memory_format == c10::MemoryFormat::Preserve) {
    if (src.is_non_overlapping_and_dense()) {
      // Copy all strides
      auto self = at::empty_strided_symint(
          src.sym_sizes(), src.sym_strides(), src.options());
      if (src._is_zerotensor()) {
        self.zero_();
      } else {
        self.copy_(src);
      }
      return self;
    } else {
      memory_format = src.suggest_memory_format();
    }
  }
  // If src is not contiguous, copy calls cnnl_contiguous to get a temp tensor
  // of src, and then copy temp to output. This will malloc a redundant memory
  // of temp tensor. Is_contiguous is more stricter than
  // is_non_overlapping_and_dense. So non_is_contiguous handles more scenarios.
  // conj and neg tensors require special handling and will be deal with in
  // copy_
  if (!src.is_neg() && !src.is_conj() && !src._is_zerotensor() &&
      !src.is_contiguous(memory_format)) {
    return cnnl_contiguous(src, memory_format);
  }
  auto self = at::empty_like(src, src.options(), memory_format);
  if (src._is_zerotensor()) {
    self.zero_();
  } else {
    self.copy_(src);
  }
  return self;
}

} // namespace ops
} // namespace torch_mlu
