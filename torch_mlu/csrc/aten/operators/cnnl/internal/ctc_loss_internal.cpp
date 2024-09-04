#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

// calculate loss and grad_probs
std::tuple<at::Tensor, at::Tensor> cnnl_ctc_loss_internal(
    const at::Tensor& probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    at::IntArrayRef il,
    at::IntArrayRef tl,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity,
    int64_t normalization) {
  int64_t batch_size = probs.size(1);
  int64_t num_labels = probs.size(2);
  TORCH_CHECK(
      (0 <= blank) && (blank < num_labels), "blank must be in label range");
  TORCH_CHECK(
      (int64_t)il.size() == batch_size,
      "input_lengths must be of size batch_size");
  TORCH_CHECK(
      (int64_t)tl.size() == batch_size,
      "target_lengths must be of size batch_size");

  // get max_target_length
  int64_t max_target_length = 0;
  for (int64_t i = 0; i < batch_size; ++i) {
    if (max_target_length < tl[i]) {
      max_target_length = tl[i];
    }
  }

  if (targets.dim() == 2) {
    TORCH_CHECK(
        targets.size(1) >= max_target_length,
        "Expected tensor to have size at least ",
        max_target_length,
        " at dimension 1, but got size ",
        targets.size(1));
  }

  // max_input_length (=T)
  int64_t max_input_length = probs.size(0);
  for (int64_t b = 0; b < batch_size; ++b) {
    TORCH_CHECK(
        il[b] <= max_input_length,
        "Expected input_lengths to have value at most ",
        max_input_length,
        ", but got value ",
        il[b]);
  }

  std::vector<cnnlCTCLossReduceMode_t> reduce_mode_list = {
      CNNL_REDUCE_MODE_NONE,
      CNNL_REDUCE_MODE_MEAN_BY_LABEL_LENGTH_AND_BATCH,
      CNNL_REDUCE_MODE_SUM,
  };
  cnnlCTCLossReduceMode_t reduce_mode = reduce_mode_list[reduction];

  at::Tensor loss;
  if (reduce_mode == CNNL_REDUCE_MODE_NONE) {
    loss = at::empty({batch_size}, probs.options());
  } else {
    loss = at::empty({}, probs.options());
  }

  std::vector<cnnlCTCLossNormalizationMode_t> norm_mode_list = {
      CNNL_NONE_NORMALIZATION,
      CNNL_SOFTMAX_NORMALIZATION,
      CNNL_LOG_SOFTMAX_NORMALIZATION,
  };
  cnnlCTCLossNormalizationMode_t norm_mode = norm_mode_list[normalization];

  cnnlCTCLossZeroInfinityMode_t zero_infinity_mode =
      zero_infinity ? CNNL_ZERO_INFINITY : CNNL_NONE_ZERO_INFINITY;

  // prepare impl
  auto probs_impl = getMluTensorImpl(probs);
  auto targets_impl = getMluTensorImpl(targets);
  auto input_lengths_impl = getMluTensorImpl(input_lengths);
  auto target_lengths_impl = getMluTensorImpl(target_lengths);
  auto loss_impl = getMluTensorImpl(loss);

  auto grad_probs = at::empty(probs.sizes(), probs.options());
  auto grad_probs_impl = getMluTensorImpl(grad_probs);

  // get current handle
  auto handle = getCurrentHandle();

  // get cnnl descriptor
  CnnlTensorDescriptor probs_desc;
  CnnlTensorDescriptor targets_desc;
  CnnlTensorDescriptor input_lengths_desc;
  CnnlTensorDescriptor target_lengths_desc;
  CnnlTensorDescriptor loss_desc;
  CnnlTensorDescriptor grad_probs_desc;
  probs_desc.set(probs, CNNL_LAYOUT_TNC);
  targets_desc.set(targets, CNNL_LAYOUT_ARRAY);
  input_lengths_desc.set(input_lengths, CNNL_LAYOUT_ARRAY);
  target_lengths_desc.set(target_lengths, CNNL_LAYOUT_ARRAY);
  loss_desc.set(loss, CNNL_LAYOUT_ARRAY);
  grad_probs_desc.set(grad_probs, CNNL_LAYOUT_TNC);

  auto probs_ptr = probs_impl->mlu_data_ptr();
  auto targets_ptr = targets_impl->mlu_data_ptr();
  auto input_lengths_ptr = input_lengths_impl->mlu_data_ptr();
  auto target_lengths_ptr = target_lengths_impl->mlu_data_ptr();
  auto loss_ptr = loss_impl->mlu_data_ptr();
  auto grad_probs_ptr = grad_probs_impl->mlu_data_ptr();

  // set ctc loss descriptor
  CnnlCTCLossDescriptor ctc_loss_desc;
  ctc_loss_desc.set(
      norm_mode,
      reduce_mode,
      zero_infinity_mode,
      blank,
      max_input_length,
      max_target_length);

  // workspace
  size_t space_size = 0;
  TORCH_CNNL_CHECK(cnnlGetCTCLossWorkspaceSize(
      handle, ctc_loss_desc.desc(), probs_desc.desc(), true, &space_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(space_size);

  // calculate
  TORCH_CNNL_CHECK(cnnlCTCLoss(
      handle,
      ctc_loss_desc.desc(),
      probs_desc.desc(),
      probs_ptr,
      targets_desc.desc(),
      targets_ptr,
      input_lengths_desc.desc(),
      input_lengths_ptr,
      target_lengths_desc.desc(),
      target_lengths_ptr,
      workspace_ptr.get(),
      space_size,
      loss_desc.desc(),
      loss_ptr,
      grad_probs_desc.desc(),
      grad_probs_ptr));

  return std::make_tuple(loss, grad_probs);
}

} // namespace ops
} // namespace torch_mlu
