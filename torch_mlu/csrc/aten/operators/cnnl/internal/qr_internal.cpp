#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

std::tuple<at::Tensor, at::Tensor> cnnl_qr_internal(
    at::Tensor& Q,
    at::Tensor& R,
    const at::Tensor& input,
    bool some) {
  // set descriptor config
  auto handle = getCurrentHandle();

  auto input_impl = getMluTensorImpl(input);
  auto descInput = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto Q_impl = getMluTensorImpl(Q);
  auto descQ = getTensorDesc(Q_impl, CNNL_LAYOUT_ARRAY);
  auto Q_ptr = mlu_data_ptr(Q_impl);

  auto R_impl = getMluTensorImpl(R);
  auto descR = getTensorDesc(R_impl, CNNL_LAYOUT_ARRAY);
  auto R_ptr = mlu_data_ptr(R_impl);

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(
      cnnlGetQRWorkspaceSize(handle, descInput.get(), some, &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlQR(
      handle,
      descInput.get(),
      input_ptr,
      descQ.get(),
      Q_ptr,
      descR.get(),
      R_ptr,
      workspace_ptr.get(),
      workspace_size,
      some));
  return std::make_tuple(Q, R);
}

} // namespace ops
} // namespace torch_mlu
