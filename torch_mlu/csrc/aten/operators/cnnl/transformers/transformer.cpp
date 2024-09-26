#include "aten/operators/cnnl/cnnl_kernel.h"

namespace torch_mlu {
namespace ops {

at::Tensor cnnl__transformer_encoder_layer_fwd(
    const at::Tensor& src,
    int64_t embed_dim,
    int64_t num_heads,
    const at::Tensor& qkv_weight,
    const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight,
    const at::Tensor& proj_bias,
    bool use_gelu,
    bool norm_first,
    double eps,
    const at::Tensor& norm_weight_1,
    const at::Tensor& norm_bias_1,
    const at::Tensor& norm_weight_2,
    const at::Tensor& norm_bias_2,
    const at::Tensor& ffn_weight_1,
    const at::Tensor& ffn_bias_1,
    const at::Tensor& ffn_weight_2,
    const at::Tensor& ffn_bias_2,
    const std::optional<at::Tensor>& mask,
    std::optional<int64_t> mask_type) {
  return at::native::transformer_encoder_layer_forward(
      src,
      embed_dim,
      num_heads,
      qkv_weight,
      qkv_bias,
      proj_weight,
      proj_bias,
      use_gelu,
      norm_first,
      eps,
      norm_weight_1,
      norm_bias_1,
      norm_weight_2,
      norm_bias_2,
      ffn_weight_1,
      ffn_bias_1,
      ffn_weight_2,
      ffn_bias_2,
      mask,
      mask_type);
}
} // namespace ops
} // namespace torch_mlu