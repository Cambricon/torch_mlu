#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
#include "c10/core/ScalarType.h"
#include "scaled_matmul_utils.h"

namespace torch_mlu {
namespace ops {

c10::MaybeOwned<Tensor> resolve_conj_if_indicated(
    const Tensor& tensor,
    bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> prepare_matrix_for_cnnl(
    const Tensor& tensor,
    bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
    transpose_tensor = !tensor.is_contiguous();
    return resolve_conj_if_indicated(tensor, true);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[1] == 1) &&
      (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if (
      (tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = false;
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> prepare_matrix_for_cnnl(
    const Tensor& tensor,
    bool& transpose_tensor,
    bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
    transpose_tensor = !tensor.is_contiguous();
    return resolve_conj_if_indicated(
        tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[1] == 1) &&
      (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if (
      (tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = false;
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}


} // namespace ops
} // namespace torch_mlu
