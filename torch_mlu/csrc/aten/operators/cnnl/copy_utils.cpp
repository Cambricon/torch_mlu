#include "aten/operators/cnnl/copy_utils.h"

namespace torch_mlu {
namespace ops {

void direct_copy_kernel_mlu(at::TensorIterator& iter) {
  auto dst = iter.tensor(0);
  auto src = iter.tensor(1);
  if (dst.dtype() == src.dtype()) {
    cnnl_copy_internal(dst, src);
  } else {
    cnnl_cast_internal(src, dst);
  }
}

void neg_conj_mlu_kernel(at::TensorIteratorBase& iter) {
  conj_mlu_kernel(iter);
  neg_mlu_kernel(iter);
}

void copy_device_to_device(
    at::TensorIterator& iter,
    bool non_blocking,
    bool p2p_enabled) {
  int64_t numel = iter.numel();
  // we can memcpy ther memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible =
      same_type && same_conj && same_neg && iter.is_contiguous();
  c10::Device dst_device = iter.device(0);
  c10::Device src_device = iter.device(1);
  torch_mlu::mlu::MLUGuard device_guard(src_device);
  auto copy_stream = getCurrentMLUStream(src_device.index());
  if (dst_device == src_device) {
    if (same_neg) {
      if (!same_conj) {
        conj_mlu_kernel(iter);
      } else {
        direct_copy_kernel_mlu(iter);
      }
    } else {
      if (!same_conj) {
        neg_conj_mlu_kernel(iter);
      } else {
        neg_mlu_kernel(iter);
      }
    }
    return;
  }
  if (memcpy_eligible && src_device != dst_device) {
    void* dst = iter.data_ptr(0);
    void* src = iter.data_ptr(1);
    size_t size = numel * iter.element_size(0);
    if (src != dst || src_device != dst_device) {
      // Allocator implement the correct call
      // (either cnrtMemcpyAsync or cnrtMemcpyPeerAsync)
      TORCH_CNRT_CHECK(MLUCachingAllocator::memcpyAsync(
          dst,
          dst_device.index(),
          src,
          src_device.index(),
          size,
          copy_stream,
          p2p_enabled));
    }
    if (!non_blocking)
      copy_stream.synchronize();
    return;
  }
  auto dst_tensor = iter.tensor(0);
  auto src_tensor = iter.tensor(1);
  auto dst_contig = dst_tensor.is_contiguous()
      ? dst_tensor
      : at::empty_like(dst_tensor, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto memory_format = dst_contig.suggest_memory_format();
  auto src_contig =
      cnnl_contiguous(src_tensor.to(iter.dtype(0)), memory_format);
  // propagate ther corrent conjugate bit.
  dst_contig._set_conj(dst_tensor.is_conj());
  src_contig._set_conj(src_tensor.is_conj());

  dst_contig._set_neg(dst_tensor.is_neg());
  src_contig._set_neg(src_tensor.is_neg());

  // perform a same-dtype copy on contiguous tensors
  TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
  TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
  dst_contig.copy_(src_contig, non_blocking);

  // if necessary, copy back into dst
  if (!dst_contig.is_same(dst_tensor)) {
    TORCH_INTERNAL_ASSERT(dst_contig.device() == dst_tensor.device());
    dst_tensor.copy_(dst_contig, non_blocking);
  }
}

} // namespace ops
} // namespace torch_mlu
