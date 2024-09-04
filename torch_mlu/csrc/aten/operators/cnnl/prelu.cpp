#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
namespace torch_mlu {
namespace ops {

at::Tensor cnnl__prelu_kernel(
    const at::Tensor& self,
    const at::Tensor& weight) {
  auto self_ = cnnl_contiguous(self);
  auto weight_ = cnnl_contiguous(weight);
  return cnnl_prelu_internal(self_, weight_);
}

std::tuple<at::Tensor, at::Tensor> cnnl__prelu_kernel_backward(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& weight) {
  auto grad_ = cnnl_contiguous(grad);
  auto self_ = cnnl_contiguous(self);
  auto weight_ = cnnl_contiguous(weight);
  return cnnl_prelu_backward_internal(grad_, self_, weight_);
}

} // namespace ops
} // namespace torch_mlu
