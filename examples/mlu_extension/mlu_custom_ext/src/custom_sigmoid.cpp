#include "custom_sigmoid.h"
#include "mlu/bang_sigmoid_sample.h"

#include "ATen/Tensor.h"
#include "torch/library.h"
#include "torch/script.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/cnnl_util.h"
#include "framework/core/MLUStream.h"
#include "framework/core/mlu_guard.h"

using namespace torch_mlu;

torch::Tensor active_sigmoid_mlu(torch::Tensor x) {
  const torch_mlu::mlu::MLUGuard device_guard(x.device());
  auto x_contiguous = torch_mlu::cnnl_contiguous(x);
  auto x_impl = getMluTensorImpl(x_contiguous);
  auto x_ptr = x_impl->mlu_data_ptr();

  auto y = at::empty_like(x_contiguous);
  auto y_contiguous = torch_mlu::cnnl_contiguous(y);
  auto y_impl = getMluTensorImpl(y_contiguous);
  auto y_ptr = y_impl->mlu_data_ptr();

  int32_t size = x_contiguous.numel();

  cnrtQueue_t stream = getCurMLUStream();
  bang_sigmoid_kernel_entry(
      stream,
      reinterpret_cast<float*>(y_ptr),
      reinterpret_cast<float*>(x_ptr),
      size);

  return y;
}

TORCH_LIBRARY_FRAGMENT(mlu_custom_ext, m) {
  m.def("active_sigmoid_mlu(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(mlu_custom_ext, PrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mlu_custom_ext::active_sigmoid_mlu"),
      TORCH_FN(active_sigmoid_mlu));
}
