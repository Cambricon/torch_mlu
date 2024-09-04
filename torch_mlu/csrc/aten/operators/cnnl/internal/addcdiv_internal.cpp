#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_addcdiv_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& alpha) {
  // get tensor impl
  auto f_alpha = alpha.toFloat();
  auto self_impl = getMluTensorImpl(self);
  auto tensor1_impl = getMluTensorImpl(tensor1);
  auto tensor2_impl = getMluTensorImpl(tensor2);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();

  // for multi input operators which are not sensitive to stride,
  // the memory format needs to be the same with the first input
  auto layout = suggest_cnnl_layout(self);

  // create the desc
  auto desc_self = getTensorDesc(self_impl, layout);
  auto desc_tensor1 = getTensorDesc(tensor1_impl, layout);
  auto desc_tensor2 = getTensorDesc(tensor2_impl, layout);
  auto desc_output = getTensorDesc(output_impl, layout);

  // get the size of workspace for brodcast and transpose
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetAddcdivWorkspaceSize_v2(
      handle,
      desc_self.get(),
      desc_tensor1.get(),
      desc_tensor2.get(),
      desc_output.get(),
      &workspace_size));
  // get the mlu ptr
  auto self_ptr = mlu_data_ptr(self_impl);
  auto tensor1_ptr = mlu_data_ptr(tensor1_impl);
  auto tensor2_ptr = mlu_data_ptr(tensor2_impl);
  auto output_ptr = mlu_data_ptr(output_impl);
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // compute ops
  TORCH_CNNL_CHECK(cnnlAddcdiv(
      handle,
      desc_self.get(),
      self_ptr,
      &f_alpha,
      desc_tensor1.get(),
      tensor1_ptr,
      desc_tensor2.get(),
      tensor2_ptr,
      workspace_ptr.get(),
      workspace_size,
      desc_output.get(),
      output_ptr));

  return output;
}

} // namespace ops
} // namespace torch_mlu
