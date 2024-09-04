#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl_prelu_internal(
    const at::Tensor& self,
    const at::Tensor& weight) {
  auto output = at::native::empty_like(self);

  int64_t weight_num = weight.numel();
  // case1: shared weight for all channels
  c10::SmallVector<int64_t, 8> cnnl_weight_size(self.dim(), 1);
  if (weight_num != 1) { // case2: multiple weights, one for each channel
    int64_t self_ndim = self.dim();
    TORCH_CHECK(self_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    if (self_ndim > 1) {
      channel_size = self.size(1); // channel is the 2nd dim of input
    }
    TORCH_CHECK(
        channel_size == weight_num,
        "Mismatch of parameter numbers and input channel size. Found "
        "parameter numbers = ",
        weight_num,
        " and channel size = ",
        channel_size,
        ".");

    // cnnlPrelu supported shape of input tensor and weight tensor are as
    // follows:
    // - input shape: [a, ..., b, ..., c], weight shape: [1, ..., 1, ..., 1]
    // - input shape: [a, ..., b, ..., c], weight shape: [1, ..., 1, ..., c]
    // - input shape: [a, ..., b, ..., c], weight shape: [1, ..., b, ..., 1]
    // only one channel of weight can > 1, so we aggregate weight_num to the 2nd
    // dim.
    cnnl_weight_size[1] = weight_num;
  }

  // get current handle
  auto handle = getCurrentHandle();

  auto self_impl = getMluTensorImpl(self);
  auto descSelf = getTensorDesc(self_impl);
  auto self_ptr = mlu_data_ptr(self_impl);

  auto weight_impl = getMluTensorImpl(weight);
  auto descWeight = getTensorDesc(
      cnnl_weight_size,
      get_contiguous_strides(cnnl_weight_size),
      getCnnlType(weight_impl),
      suggestCnnlLayout(weight_impl));
  auto weight_ptr = mlu_data_ptr(weight_impl);

  auto output_impl = getMluTensorImpl(output);
  auto descOutput = getTensorDesc(output_impl);
  auto output_ptr = mlu_data_ptr(output_impl);

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlPrelu(
      handle,
      descSelf.get(),
      self_ptr,
      descWeight.get(),
      weight_ptr,
      descOutput.get(),
      output_ptr));
  return output;
}

std::tuple<at::Tensor, at::Tensor> cnnl_prelu_backward_internal(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& weight) {
  std::vector<int64_t> tmp = at::infer_size(grad.sizes(), self.sizes());
  std::vector<int64_t> grad_input_dims = at::infer_size(tmp, weight.sizes());
  auto grad_input = at::empty(grad_input_dims, self.options());
  auto grad_weight = at::native::empty_like(grad_input);

  // get current handle
  auto handle = getCurrentHandle();

  auto input_impl = getMluTensorImpl(self);
  auto descInput = getTensorDesc(input_impl, CNNL_LAYOUT_ARRAY);
  auto input_ptr = mlu_data_ptr(input_impl);

  auto grad_impl = getMluTensorImpl(grad);
  auto descGrad = getTensorDesc(grad_impl, CNNL_LAYOUT_ARRAY);
  auto grad_ptr = mlu_data_ptr(grad_impl);

  auto weight_impl = getMluTensorImpl(weight);
  auto descWeight = getTensorDesc(weight_impl, CNNL_LAYOUT_ARRAY);
  auto weight_ptr = mlu_data_ptr(weight_impl);

  auto grad_input_impl = getMluTensorImpl(grad_input);
  auto descOutputGrad = getTensorDesc(grad_input_impl, CNNL_LAYOUT_ARRAY);
  auto grad_input_ptr = mlu_data_ptr(grad_input_impl);

  auto grad_weigh_impl = getMluTensorImpl(grad_weight);
  auto descWeightGrad = getTensorDesc(grad_weigh_impl, CNNL_LAYOUT_ARRAY);
  auto grad_weight_ptr = mlu_data_ptr(grad_weigh_impl);

  // workspace
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetPreluBackwardV2WorkspaceSize(
      handle,
      descInput.get(),
      descGrad.get(),
      descWeight.get(),
      descOutputGrad.get(),
      descWeightGrad.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  // malloc mlu memory

  // set descriptor config
  TORCH_CNNL_CHECK(cnnlPreluBackwardV2(
      handle,
      descInput.get(),
      input_ptr,
      descGrad.get(),
      grad_ptr,
      descWeight.get(),
      weight_ptr,
      workspace_ptr.get(),
      workspace_size,
      descOutputGrad.get(),
      grad_input_ptr,
      descWeightGrad.get(),
      grad_weight_ptr));
  return std::make_tuple(grad_input, grad_weight);
}

} // namespace ops
} // namespace torch_mlu
