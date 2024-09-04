#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "aten/utils/dispatch.h"
#include <ATen/native/SortingUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/MemoryOverlap.h>

using at::maybe_wrap_dim;
using at::native::_reduction_with_indices_allocate_or_resize_output;
using at::native::zero_numel_check_dims;

namespace torch_mlu {
namespace ops {

std::tuple<Tensor&, Tensor&> kthvalue_out_impl_mlu(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);
  zero_numel_check_dims(self, dim, "kthvalue()");

  TORCH_CHECK(
      k >= 1 && k <= slicesize,
      "kthvalue(): selected number k out of range for dimension ",
      dim);

  at::assert_no_overlap(self, values);

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  auto memory_format = self.suggest_memory_format();
  auto self_contiguous = cnnl_contiguous(self, memory_format);
  auto values_contiguous = cnnl_contiguous(values, memory_format);
  auto indices_contiguous = cnnl_contiguous(indices, memory_format);

  if (self.numel() != 0) {
    AT_DISPATCH_MLU_FLOATING_TYPES_HALF_AND_BFLOAT16(
        self_contiguous.scalar_type(), "cnnl_kthvalue", [&] {
          indices_contiguous = cast_long_to_int_if_needed(indices_contiguous);
          cnnl_kthvalue_internal(
              values_contiguous, indices_contiguous, self_contiguous, k, dim);
        });

    if (is_copy_necessary(values, values_contiguous)) {
      values.copy_(values_contiguous);
    }
    if (is_copy_necessary(indices, indices_contiguous)) {
      indices.copy_(indices_contiguous);
    }
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
  return std::tuple<at::Tensor&, at::Tensor&>(values, indices);
}

std::tuple<at::Tensor&, at::Tensor&> cnnl_kthvalue_out(
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    at::Tensor& values,
    at::Tensor& indices) {
  auto result = [&]() {
    at::NoNamesGuard guard;
    return kthvalue_out_impl_mlu(values, indices, self, k, dim, keepdim);
  }();
  at::namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  at::namedinference::propagate_names_for_reduction(
      indices, self, dim, keepdim);
  return result;
}

} // namespace ops
} // namespace torch_mlu
