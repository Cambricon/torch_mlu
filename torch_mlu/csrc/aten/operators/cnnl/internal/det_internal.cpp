#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_det_internal(
    at::Tensor& output,
    const at::Tensor& input,
    cnnlDetMode_t mode) {
  CnnlTensorDescriptor descInput;
  CnnlTensorDescriptor descOutput;
  descInput.set(input);
  descOutput.set(output);
  // malloc mlu memory
  auto input_impl = getMluTensorImpl(input);
  auto output_impl = getMluTensorImpl(output);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto output_ptr = output_impl->mlu_data_ptr();
  // set descriptor config
  auto handle = getCurrentHandle();
  size_t ws_size = 0;
  TORCH_CNNL_CHECK(cnnlGetDetWorkspaceSize(
      /* handle     */ handle,
      /* input_desc */ descInput.desc(),
      /* size       */ &ws_size));
  auto ws_ptr = torch_mlu::MLUCachingAllocator::get()->allocate(ws_size);

  TORCH_CNNL_CHECK(cnnlDet_v2(
      /* handle         */ handle,
      /* mode           */ mode,
      /* input_desc     */ descInput.desc(),
      /* input          */ input_ptr,
      /* workspace      */ ws_ptr.get(),
      /* workspace_size */ ws_size,
      /* output_desc    */ descOutput.desc(),
      /* output         */ output_ptr,
      /* sign_desc      */ nullptr,
      /* sign           */ nullptr));

  return output;
}

} // namespace ops
} // namespace torch_mlu
