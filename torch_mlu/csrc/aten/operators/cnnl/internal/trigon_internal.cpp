#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor& cnnl_trigon_internal(
    at::Tensor& output,
    const at::Tensor& self,
    cnnlTrigonFunctionMode_t mode) {
  if (self.numel() == 0) {
    return output;
  }

  auto input_impl = getMluTensorImpl(self);
  auto output_impl = getMluTensorImpl(output);

  // get current handle
  auto handle = getCurrentHandle();
  CnnlTrigonDescriptor trigon_desc;
  trigon_desc.set(mode, CNNL_COMPUTATION_HIGH_PRECISION);

  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // set descriptor config
  if (mode == CNNL_TRIGON_COS) {
    TORCH_CNNL_CHECK(cnnlCos_v2(
        handle,
        CNNL_COMPUTATION_HIGH_PRECISION,
        input_desc.get(),
        input_ptr,
        output_desc.get(),
        output_ptr));
  } else if (mode == CNNL_TRIGON_SIN) {
    TORCH_CNNL_CHECK(cnnlSin_v2(
        handle,
        CNNL_COMPUTATION_HIGH_PRECISION,
        input_desc.get(),
        input_ptr,
        output_desc.get(),
        output_ptr));
  } else {
    TORCH_CNNL_CHECK(cnnlTrigonForward(
        handle,
        trigon_desc.desc(),
        input_desc.get(),
        input_ptr,
        output_desc.get(),
        output_ptr));
  }
  return output;
}

at::Tensor& cnnl_atan2_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& other) {
  // get current handle
  auto handle = getCurrentHandle();

  auto input_impl = getMluTensorImpl(input);
  auto other_impl = getMluTensorImpl(other);
  auto output_impl = getMluTensorImpl(output);

  auto input_desc = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto other_desc = getTensorDesc(other_impl, CNNL_LAYOUT_ARRAY);
  auto output_desc = getTensorDesc(output_impl, CNNL_LAYOUT_ARRAY);

  // malloc mlu memory
  auto input_ptr = mlu_data_ptr(input_impl);
  auto other_ptr = mlu_data_ptr(other_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  // get the size of workspace for brodcast
  size_t workspace_size;
  TORCH_CNNL_CHECK(cnnlGetAtan2WorkspaceSize(
      handle,
      input_desc.get(),
      other_desc.get(),
      output_desc.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlAtan2(
      handle,
      prefer,
      input_desc.get(),
      input_ptr,
      other_desc.get(),
      other_ptr,
      workspace_ptr.get(),
      workspace_size,
      output_desc.get(),
      output_ptr));
  return output;
}

} // namespace ops
} // namespace torch_mlu
