#include "aten/utils/dispatch.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"
namespace torch_mlu {
namespace ops {

at::Tensor cnnl__masked_softmax(
    const at::Tensor& self,
    const at::Tensor& mask,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> mask_type) {
  TORCH_CHECK(
      mask.scalar_type() == at::ScalarType::Bool,
      "Mask should be a boolean tensor");

  TORCH_CHECK(mask_type.has_value(), "Mask Type should be defined");
  int64_t mask_type_ = mask_type.value();
  TORCH_CHECK(
      (mask_type_ == 0) || (mask_type_ == 1) || (mask_type_ == 2),
      "Mask Type should be 0 (src_mask), 1 (src_key_padding_mask), or 2 (default_mask)");

  // If self is [B, H, T, T] and mask is [T, T]
  // expand mask to [B, H, T, T] and treat it like regular mask
  // mask_type == 0 => mask is a src_mask
  bool is_TxT_mask = (mask_type == 0) && self.dim() == 4 && mask.dim() == 2 &&
      self.size(3) == mask.size(1) && self.size(2) == mask.size(0) &&
      mask.size(0) == mask.size(1);

  // self is [B, H, T, T] and mask is [B, T]
  // mask_type == 1 => mask is a key_padding_mask
  bool is_BxT_mask = (mask_type == 1) &&
      (self.dim() == 4 && mask.dim() == 2 && self.size(0) == mask.size(0) &&
       self.size(2) == mask.size(1) && self.size(3) == mask.size(1));

  // If mask_type == 2, then mask.sizes() must equal self.sizes()
  TORCH_CHECK(
      mask.sizes() == self.sizes() || is_BxT_mask || is_TxT_mask,
      "Mask shape should match input. mask: ",
      mask.sizes(),
      " input: ",
      self.sizes());

  auto input = self.dim() == 0 ? self.view(1) : self;
  auto mask_ = mask.dim() == 0 ? mask.view(1) : mask;

  if (is_TxT_mask) {
    mask_ = mask_.expand(input.sizes());
  } else if (is_BxT_mask) {
    // cnnl-extra scale limitation:
    // The third dimension from the end can be 1. For example, when the shape of
    // self is [B, H, T, T], the shape of mask can be [B, 1, T, T].
    mask_ = mask_.view({input.size(0), 1, 1, input.size(2)})
                .expand({input.size(0), 1, input.size(2), input.size(3)});
  }
  auto mask_con = cnnl_contiguous(mask_, c10::MemoryFormat::Contiguous);

  int64_t dim_ = dim.has_value() ? dim.value() : input.dim() - 1;

  auto input_con = cnnl_contiguous(input, c10::MemoryFormat::Contiguous);
  at::Tensor output = at::empty_like(input_con, input_con.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "masked_softmax",
      [&] {
        cnnl_masked_softmax_internal(
            output,
            input_con,
            mask_con,
            dim_,
            CNNL_MASKED_SOFTMAX_MASKED_FILL_NEG_INF);
      });

  return output;
}

} // namespace ops
} // namespace torch_mlu