#pragma once

#include <optional>

#include <ATen/AccumulateType.h>
#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/layer_norm.h>
#include <c10/core/DynamicCast.h>

#include "aten/cnnl/cnnlDescriptors.h"
#include "aten/cnnl/cnnlTensorDesc.h"
#include "aten/cnnl/cnnlHandle.h"
#include "aten/cnnl/cnnlHeuristicResult.h"
#include "aten/cnnl/cnnlAlgorithms.h"
#include "aten/utils/binaryops_util.h"
#include "aten/utils/cnnl_util.h"
#include "aten/utils/exceptions.h"
#include "aten/utils/internal_util.h"
#include "aten/utils/tensor_util.h"
#include "aten/utils/utils.h"
#include "aten/utils/accumulate_type.h"
#include "framework/core/MLUStream.h"
#include "framework/core/tensor_impl.h"
#include "aten/utils/types.h"
#include "utils/cnlog.h"
#include "utils/common.h"

using at::IntArrayRef;
using at::Tensor;
using at::TensorList;

enum class ReduceType {
  Reduce_Sum,
  Reduce_Mean,
  Reduce_Max,
  Reduce_Min,
  Reduce_Any,
  Reduce_And,
  Reduce_All,
  Reduce_Mul,
  Reduce_Norm1,
  Reduce_Norm2,
  Reduce_NormP
};

enum class CumType {
  Cum_Min,
  Cum_Max,
};

enum class LogaddexpType { LOGADDEXP_BASE_E, LOGADDEXP_BASE_2 };

namespace torch_mlu {
namespace ops {

static inline std::pair<tensorDescPtr_t, void*> GetTensorDescAndMluDataPtr(
    const at::Tensor& t) {
  auto impl = getMluTensorImpl(t);
  void* tensor_ptr = mlu_data_ptr(impl);
  auto desc = getTensorDesc(impl);
  return std::make_pair<tensorDescPtr_t, void*>(
      std::move(desc), std::move(tensor_ptr));
};

static inline std::pair<tensorDescPtr_t, void*> GetTensorDescAndMluDataPtr(
    const at::Tensor& t,
    const cnnlTensorLayout_t layout) {
  auto impl = getMluTensorImpl(t);
  void* tensor_ptr = mlu_data_ptr(impl);
  auto desc = getTensorDesc(impl, layout);
  return std::make_pair<tensorDescPtr_t, void*>(
      std::move(desc), std::move(tensor_ptr));
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_cell_forward_internal(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& input_bias,
    const at::Tensor& hidden_bias,
    const at::Tensor& cx,
    const at::Tensor& hy,
    const at::Tensor& cy,
    at::Tensor& workspace);

std::tuple<at::Tensor, at::Tensor> lstm_cell_backward_internal(
    const at::Tensor& grad_hy,
    const at::Tensor& grad_cy,
    const at::Tensor& cx,
    const at::Tensor& cy,
    const at::Tensor& workspace,
    const at::Tensor& grad_gates,
    const at::Tensor& grad_cx);

std::tuple<at::Tensor, at::Tensor> gru_cell_forward_internal(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& input_bias,
    const at::Tensor& hidden_bias,
    const at::Tensor& hx,
    const at::Tensor& hy,
    at::Tensor& workspace);

std::tuple<at::Tensor, at::Tensor, at::Tensor> gru_cell_backward_internal(
    const at::Tensor& grad_hy,
    const at::Tensor& workspace,
    const at::Tensor& grad_input_gates,
    const at::Tensor& grad_hidden_gates,
    const at::Tensor& grad_hx);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl_rnn_training_internal(
    const at::Tensor& input,
    const at::Tensor& hx,
    const at::Tensor& cx,
    TensorList params,
    bool has_biases,
    cnnlRNNMode_t mode,
    int64_t hidden_size,
    int64_t proj_size,
    int64_t num_layers,
    bool bidirectional,
    bool train,
    at::IntArrayRef batch_sizes);

std::tuple<Tensor, Tensor, Tensor> cnnl_rnn_backward_input_internal(
    const Tensor& input_r,
    const Tensor& weight_buf,
    int64_t weight_stride0,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& output_r,
    const Tensor& grad_output_r,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t proj_size,
    int64_t fn_num_layers,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    const int* batch_sizes_int_ptr,
    const at::Tensor& dev_batch_sizes,
    const Tensor& fn_dropout_state,
    const Tensor& fn_reserve,
    std::array<bool, 3> output_mask);

void cnnl_native_batch_norm_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& scale_t,
    const at::Tensor& bias_t,
    const at::Tensor& moving_mean,
    const at::Tensor& moving_var,
    Tensor& saved_mean,
    Tensor& saved_invstd,
    bool training,
    double momentum,
    double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl_native_batch_norm_backward_internal(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_invstd,
    bool training,
    double eps);

std::tuple<at::Tensor, at::Tensor> cnnl_batch_norm_stats_internal(
    const at::Tensor& input,
    float epsilon);

std::tuple<at::Tensor, at::Tensor>
cnnl_batch_norm_gather_stats_with_counts_internal(
    const at::Tensor& mean_all,
    const at::Tensor& invstd_all,
    const at::Tensor& count_all,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    float momentum,
    float epsilon);

at::Tensor& cnnl_batch_norm_elemt_out_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& mean,
    const at::Tensor& invstd);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl_batch_norm_backward_reduce_internal(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const at::Tensor& weight,
    const bool input_g,
    const bool weight_g,
    const bool bias_g);

at::Tensor cnnl_batch_norm_backward_elemt_internal(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
    const at::Tensor& weight,
    const at::Tensor& sum_dy,
    const at::Tensor& sum_dy_xmu,
    const at::Tensor& count);

at::Tensor& cnnl_constant_pad_nd_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int pad[][2],
    at::Scalar value_scalar,
    c10::MemoryFormat memory_format);

void cnnl_upsample_internal(
    const at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    bool align_center,
    cnnlInterpMode_t interp_mode,
    std::optional<double> scales_h = std::nullopt,
    std::optional<double> scales_w = std::nullopt,
    std::optional<double> scales_d = std::nullopt,
    bool is_exact = false);

void cnnl_upsample_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    bool align_corners,
    bool align_center,
    cnnlInterpMode_t interp_mode,
    std::optional<double> scales_h = std::nullopt,
    std::optional<double> scales_w = std::nullopt,
    std::optional<double> scales_d = std::nullopt);

void cnnl_mse_loss_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction);

void cnnl_mse_loss_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction);

void cnnl_smooth_l1_loss_forward_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double beta);

void cnnl_smooth_l1_loss_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& target,
    int64_t reduction,
    double beta);

at::Tensor cnnl_expand_internal(
    const at::Tensor& self,
    at::IntArrayRef size,
    bool implicit);

at::Tensor cnnl_expand_out_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef size);
at::Tensor& cnnl_copy_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor cnnl_prelu_internal(
    const at::Tensor& self,
    const at::Tensor& weight);

std::tuple<at::Tensor, at::Tensor> cnnl_prelu_backward_internal(
    const at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& weight);

void cnnl_multinomial_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    std::optional<at::Generator> gen);

at::Tensor cnnl_permute_internal(const at::Tensor& input, at::IntArrayRef dims);

at::Tensor& cnnl_permute_out_internal(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef dims);

at::Tensor& cnnl_as_strided_backward_internal(
    at::Tensor& grad_y,
    const at::Tensor& grad_x,
    at::IntArrayRef stride,
    std::optional<int64_t> storage_offset);

at::Tensor d2d_copy_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor cnnl_transpose_internal(
    const at::Tensor& self,
    int64_t dim0,
    int64_t dim1);

void cnnl_topk_internal(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    bool stable);

void cnnl_median_internal(
    const at::Tensor& input,
    int64_t dim,
    at::Tensor& values,
    at::Tensor& indices,
    bool is_dim_none,
    bool ignore_nan);

void cnnl_addmm_out_internal(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    bool is_trans_self_,
    bool is_trans_mat1_,
    bool is_trans_mat2_,
    const at::Scalar& beta_,
    const at::Scalar& alpha_,
    bool allow_tf32_);

void cnnl_addmm_bias_out_internal(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    bool is_trans_self_,
    bool is_trans_mat1_,
    bool is_trans_mat2_,
    const at::Scalar& beta_,
    const at::Scalar& alpha_,
    cnnlActivationMode_t mode,
    bool allow_tf32_);

void cnnl_addmm_sparse_out_internal(
    at::Tensor& result,
    const at::Tensor& mat1_row_indices,
    const at::Tensor& mat1_col_indices,
    const at::Tensor& mat1_values,
    const at::Tensor& mat2,
    int nnz,
    int m,
    int n,
    int k,
    bool is_trans_mat1_,
    bool is_trans_mat2_,
    const at::Scalar& beta_,
    const at::Scalar& alpha_,
    bool allow_tf32_);

void cnnl_baddbmm_out_internal(
    bool transa,
    bool transb,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t batch_size,
    const at::Tensor& self,
    int32_t ldc,
    int64_t stride_c,
    const at::Scalar& alpha,
    const at::Tensor& batch1,
    int32_t lda,
    int64_t stride_a,
    const at::Tensor& batch2,
    int32_t ldb,
    int64_t stride_b,
    const at::Scalar& beta,
    at::Tensor& result,
    int32_t ldd,
    int64_t stride_d,
    bool allow_tf32_);

void cnnl_cast_internal(const at::Tensor& input, at::Tensor& output);

void cnnl_cat_internal(
    const at::ITensorListRef& inputs,
    const at::Tensor& output,
    const int64_t dim,
    const c10::MemoryFormat memory_format);

void cnnl_scatter_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src,
    cnnlScatterMode_t mode);

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
    int64_t normalization);

void cnnl_exponential_internal(
    at::Tensor& output,
    double lambda,
    std::optional<at::Generator> generator);

at::Tensor cnnl_max_unpool2d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding);

at::Tensor cnnl_max_unpool2d_backward_internal(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& grad_input);

void cnnl_reflection_pad2d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef padding);

void cnnl_reflection_pad1d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef padding);

void cnnl_reflection_pad2d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef padding);

void cnnl_reflection_pad1d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef padding);

at::Tensor& cnnl_neg_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_unfold_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dimension,
    int64_t size,
    int64_t step);

at::Tensor cnnl_slice_internal(
    const at::Tensor& input,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step);

at::Tensor cnnl_multi_dims_slice_internal(
    const at::Tensor& input,
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& steps);
at::Tensor& cnnl_diag_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t k);

at::Tensor& cnnl_fill_internal(at::Tensor& input, const at::Tensor& value);

at::Tensor& cnnl_fill_internal(at::Tensor& input, const at::Scalar& other);

at::Tensor& cnnl_transform_out_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Scalar alpha_scalar,
    const at::Scalar beta_scalar);

at::Tensor cnnl_optensor_out_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha_scalar1,
    at::Scalar alpha_scalar2,
    at::Scalar beta_scalar,
    cnnlOpTensorDesc_t op_type);

at::Tensor cnnl_optensor_out_with_scalar_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Scalar alpha_scalar1,
    at::Scalar alpha_scalar2,
    at::Scalar beta_scalar,
    cnnlOpTensorDesc_t op_type);

at::Tensor& cnnl_angle_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_abs_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_normal_internal(
    at::Tensor& output,
    double mean,
    double std,
    std::optional<at::Generator> gen);

at::Tensor& cnnl_index_select_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index);
at::Tensor& cnnl_masked_fill_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& mask,
    const at::Tensor& value_tensor);

void cnnl_round_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_roll_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<int64_t>& shifts,
    const std::vector<int64_t>& dims);

void cnnl_clamp_internal(
    at::Tensor& output,
    const at::Tensor& self,
    at::optional<at::Scalar> min,
    at::optional<at::Scalar> max);

void cnnl_clamp_tensor_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& min,
    const at::Tensor& max);

void cnnl_softplus_internal(
    at::Tensor& output,
    const at::Tensor& input,
    at::Scalar beta,
    at::Scalar threshold);

void cnnl_softplus_backward_internal(
    const at::Tensor& self,
    const at::Tensor& grad_output,
    at::Scalar beta,
    at::Scalar threshold,
    at::Tensor& result);

void cnnl_activation_internal(
    at::Tensor& output,
    const at::Tensor& input,
    cnnlActivationMode_t mode,
    /*only for glu*/ int64_t dim = 0,
    /*leaky-relu, elu, silu*/ const at::Scalar& negative_slope = 0.0,
    /*only for elu, silu*/ const at::Scalar& scale = 0.0,
    /*only for elu, silu*/ const at::Scalar& input_scale = 0.0,
    /*only for gelu*/ bool approximate = true);

void cnnl_activation_backward_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& grad,
    cnnlActivationMode_t mode,
    /*only for glu*/ int64_t dim = 0,
    /*leaky-relu, elu, silu*/ const at::Scalar& negative_slope = 0.0,
    /*only for elu, silu*/ const at::Scalar& scale = 0.0,
    /*only for elu, silu*/ const at::Scalar& input_scale = 0.0,
    /*only for elu backward*/ bool is_result = false,
    /*only for gelu*/ bool approximate = true);

void cnnl_maximum_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& other);

void cnnl_minimum_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& other);

void cnnl_hardtanh_backward_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& grad,
    at::Scalar min_val,
    at::Scalar max_val);

void cnnl_threshold_internal(
    at::Tensor& output,
    const at::Tensor& input,
    at::Scalar threshold,
    at::Scalar value);

void cnnl_threshold_backward_internal(
    at::Tensor& result,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    at::Scalar threshold);

void cnnl_logic_internal(
    at::Tensor& output,
    const at::Tensor& input_,
    const at::Tensor& other_,
    cnnlLogicOp_t logic_type,
    const at::ScalarType& compute_dtype);

void cnnl_logic_not_internal(at::Tensor& output, const at::Tensor& input);

void cnnl_kthvalue_internal(
    at::Tensor& values,
    at::Tensor& indices,
    const at::Tensor& self,
    int64_t k,
    int64_t dim);

at::Tensor& cnnl_erfinv_internal(at::Tensor& output, const at::Tensor& self);

at::Tensor& cnnl_exp_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_expm1_internal(at::Tensor& output, const at::Tensor& input);

void cnnl_softmax_out_internal(
    const at::Tensor& input,
    int64_t dim,
    at::Tensor& output,
    cnnlSoftmaxAlgorithm_t algo);

void cnnl_softmax_backward_out_internal(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::Tensor& grad_input,
    cnnlSoftmaxAlgorithm_t algo);

at::Tensor& cnnl_log_internal(
    at::Tensor& output,
    const at::Tensor& input,
    cnnlLogBase_t base);

at::Tensor& cnnl_log1p_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_reciprocal_internal(
    at::Tensor& output,
    const at::Tensor& input);

at::Tensor& cnnl_sqrt_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_rsqrt_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_index_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const std::vector<at::Tensor>& indices);

void cnnl_index_add_internal(
    const at::Tensor& output,
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source);

at::Tensor& cnnl_pow_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const std::optional<at::Tensor>& exponent_t = std::nullopt,
    const std::optional<at::Scalar>& exponent_s = std::nullopt);

at::Tensor cnnl_pool2d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    bool ceil_mode,
    bool count_include_pad,
    int64_t pool_mode_row,
    int dilationH = 1,
    int dilationW = 1);

at::Tensor cnnl_pool2d_backward_internal(
    const at::Tensor& gradInput,
    const at::Tensor& gradOutput,
    const at::Tensor& self,
    const at::Tensor& index,
    const int64_t kH,
    const int64_t kW,
    const int64_t dH,
    const int64_t dW,
    const int64_t padH,
    const int64_t padW,
    bool ceil_mode,
    bool count_include_pad,
    int dilationH = 1,
    int dilationW = 1);

at::Tensor cnnl_pool3d_internal(
    const at::Tensor& output,
    const at::Tensor& self,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool ceil_mode,
    bool count_include_pad,
    int64_t pool_mode_row,
    int dilationT = 1,
    int dilationH = 1,
    int dilationW = 1);

at::Tensor cnnl_pool3d_backward_internal(
    const at::Tensor& gradInput,
    const at::Tensor& gradOutput,
    const at::Tensor& self,
    const at::Tensor& indices,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool ceil_mode,
    bool count_include_pad,
    int dilaitonT = 1,
    int dilationH = 1,
    int dilationW = 1);

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool2d_with_indices_internal(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    bool ceil_mode,
    int dilationH = 1,
    int dilationW = 1);

std::tuple<at::Tensor, at::Tensor> cnnl_max_pool3d_with_indices_internal(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& self,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool ceil_mode,
    int dilationT = 1,
    int dilationH = 1,
    int dilationW = 1);

void cnnl_adaptive_avg_pool_internal(
    at::Tensor& output,
    const at::Tensor& input,
    at::IntArrayRef output_size);

void cnnl_adaptive_avg_pool_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input);

void cnnl_adaptive_max_pool2d_internal(
    at::Tensor& output,
    at::Tensor& indices,
    const at::Tensor& input,
    at::IntArrayRef output_size);

void cnnl_adaptive_max_pool2d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& indices);

at::Tensor cnnl_where_internal(
    at::Tensor& output,
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other);

at::Tensor& cnnl_nonzero_internal(at::Tensor& out, const at::Tensor& self);

at::Tensor& cnnl_index_put_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const std::vector<at::Tensor>& indices,
    const at::Tensor& value,
    bool accumulate);

at::Tensor& cnnl_arange_internal(
    at::Tensor& out,
    const void* start_ptr,
    const void* step_ptr);

void cnnl_embedding_internal(
    const at::Tensor& weight,
    const at::Tensor& indices,
    at::Tensor& output,
    int64_t padding_idx,
    bool scale_grad_by_freq);

at::Tensor cnnl_nms_internal(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold);

void cnnl_embedding_dense_backward_internal(
    const at::Tensor& grad,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    at::Tensor& output);

void cnnl_embedding_bag_internal(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& per_sample_weights,
    at::Tensor& output,
    at::Tensor& offset2bag,
    at::Tensor& bag_size,
    at::Tensor& max_indices,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    bool include_last_offset,
    int64_t padding_idx);

void cnnl_embedding_bag_dense_backward_internal(
    const at::Tensor& grad,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const at::Tensor& offset2bag,
    const at::Tensor& bag_size,
    const at::Tensor& per_sample_weights,
    const at::Tensor& max_indices,
    at::Tensor& output,
    int64_t mode);

at::Tensor& cnnl_masked_scatter_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& mask,
    const at::Tensor& source);

at::Tensor& cnnl_grid_sampler_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool is_3d,
    bool align_corners);

void cnnl_grid_sampler_backward_internal(
    at::Tensor& grad_input,
    at::Tensor& grad_grid,
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool is_3d,
    bool align_corners);

void cnnl_native_layer_norm_internal(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& output,
    at::Tensor& mean,
    at::Tensor& rstd,
    double eps,
    int64_t axis);

void cnnl_native_layer_norm_backward_internal(
    const at::Tensor& diff_z,
    const at::Tensor& x,
    int64_t axis,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& weight,
    at::Tensor& diff_x,
    at::Tensor& diff_weight,
    at::Tensor& diff_bias);

void cnnl_ceil_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor& cnnl_boxes_overlap_bev_out_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other);

at::Tensor& cnnl_div_out_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other,
    const std::string& rounding_mode);

at::Tensor& cnnl_floor_internal(at::Tensor& output, const at::Tensor& self);

at::Tensor& cnnl_trunc_internal(at::Tensor& output, const at::Tensor& self);

at::Tensor& cnnl_repeat_internal(at::Tensor& output, const at::Tensor& input);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_group_norm_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    at::Tensor& mean,
    at::Tensor& rstd,
    double eps,
    int64_t num_groups);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_group_norm_backward_internal(
    const at::Tensor& x_contiguous,
    const at::Tensor& dY,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& gamma,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    at::Tensor& dX,
    at::Tensor& dgamma,
    at::Tensor& dbeta);

at::Tensor cnnl_gather_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index);

at::Tensor& cnnl_uniform_internal(
    at::Tensor& self,
    std::optional<at::Generator> gen,
    float min,
    float max);

at::Tensor& cnnl_convolution_forward_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const int64_t* padding,
    const int64_t* stride,
    const int64_t* dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32,
    bool is_depth_wise_conv);

at::Tensor& cnnl_convolution_backward_input_internal(
    at::Tensor& input_grad,
    const at::Tensor& output_grad,
    const at::Tensor& weight,
    const int64_t* stride,
    const int64_t* padding,
    const int64_t* dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

at::Tensor& cnnl_convolution_backward_weight_internal(
    at::Tensor& grad_weight,
    const at::Tensor& output_grad,
    const at::Tensor& input,
    const int64_t* stride,
    const int64_t* padding,
    const int64_t* dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32);

at::Tensor& cnnl_bias_backward_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim);

at::Tensor& cnnl_random_internal(
    at::Tensor& output,
    int64_t from,
    int64_t to,
    std::optional<at::Generator> gen);

void fused_dropout_internal(
    at::Tensor& output,
    at::Tensor& mask,
    const at::Tensor& self,
    double p,
    std::optional<at::Generator> gen);

at::Tensor& cnnl_nan_to_num_internal(
    at::Tensor& output,
    const at::Tensor& input,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf);

at::Tensor& cnnl_masked_scale_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& mask,
    const float scale);

void cnnl_index_fill_internal(
    at::Tensor& output,
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    at::Scalar value);

void cnnl_fmod_internal(
    at::Tensor& result,
    const at::Tensor& self,
    const at::Tensor& other);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_unique_internal(
    const at::Tensor& self,
    int64_t dim,
    bool return_inverse,
    bool return_counts);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_unique_consecutive_internal(
    const at::Tensor& self,
    int64_t dim,
    bool return_inverse,
    bool return_counts);

at::Tensor cnnl_put_internal(
    at::Tensor& output_,
    const at::Tensor& self_,
    const at::Tensor& index_,
    const at::Tensor& source_,
    const bool accumulate);

void cnnl_erf_internal(at::Tensor& output, const at::Tensor& self);

void cnnl_erfc_internal(at::Tensor& output, const at::Tensor& self);

void cnnl_bce_internal(
    at::Tensor& loss,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction);

void cnnl_bce_bp_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction);

void cnnl_bce_with_logits_internal(
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    const at::Tensor& pos_weight,
    int64_t reduction,
    at::Tensor& output);

void cnnl_nll_loss_forward_internal(
    at::Tensor& output,
    at::Tensor& total_weight,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index);

void cnnl_nll_loss_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const at::Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight);

at::Tensor& cnnl_bitwise_op_out_internal(
    at::Tensor& out,
    const at::Tensor& self,
    const at::Tensor& other,
    const cnnlBitComputeOp_t& op_type);

at::Tensor& cnnl_sign_internal(at::Tensor& output, const at::Tensor& input);

void cnnl_index_copy_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source);

std::tuple<at::Tensor, at::Tensor> cnnl_qr_internal(
    at::Tensor& Q,
    at::Tensor& R,
    const at::Tensor& input,
    bool some);

at::Tensor& cnnl_replication_pad1d_backward_internal(
    at::Tensor& grad_input,
    const at::Tensor& grad_output,
    at::IntArrayRef padding);

void cnnl_replication_pad2d_internal(
    const at::Tensor& output,
    const at::Tensor& self,
    at::IntArrayRef padding);

at::Tensor& cnnl_replication_pad2d_backward_internal(
    at::Tensor& grad_output,
    const at::Tensor& grad_input,
    at::IntArrayRef padding);

at::Tensor& cnnl_addcdiv_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& alpha);

at::Tensor& cnnl_addcmul_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& alpha);

void cnnl_histc_internal(
    const at::Tensor self,
    int64_t bins,
    float minvalue,
    float maxvalue,
    const at::Tensor output);

at::Tensor cnnl_dcn_forward_internal(
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const std::vector<int64_t>& output_sizes,
    int* padding,
    int* stride,
    int* dilation,
    int64_t deformable_group,
    int64_t conv_group,
    int64_t im2col_step,
    int bitwidth,
    bool use_mask);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl_dcn_backward_internal(
    const at::Tensor& grad,
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& weight,
    const at::Tensor& bias,
    int* padding,
    int* stride,
    int* dilation,
    int64_t deformable_group,
    int64_t conv_group,
    int64_t im2col_step,
    std::array<bool, 2> grad_input_mask,
    bool use_mask,
    int bitwidth);

void cnnl_reduce_internal(
    const at::Tensor& input,
    at::Tensor& output,
    at::Tensor& index,
    const std::vector<int64_t>& reduce_dim,
    cnnlReduceOp_t reduce_mode,
    const cnnlReduceIndices_t reduce_indices,
    float norm_p = 0.);

void cnnl_var_internal(
    const at::Tensor& self,
    at::Tensor& output,
    at::IntArrayRef dim,
    double correction_value);

void cnnl_std_internal(
    const at::Tensor& self,
    at::Tensor& output,
    at::IntArrayRef dim,
    double correction_value);

void cnnl_remainder_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& other);

void cnnl_cummin_max_internal(
    at::Tensor& input,
    at::Tensor& values,
    at::Tensor& indices,
    int64_t dim,
    CumType kind);

void cnnl_cumsum_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim);

void cnnl_cumprod_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t dim);

void cnnl_flip_internal(
    const at::Tensor& self,
    at::Tensor& output,
    at::IntArrayRef dims);

at::Tensor cnnl_take_internal(
    const at::Tensor& input_,
    const at::Tensor& index_,
    at::Tensor& output);

at::Tensor& cnnl_trigon_internal(
    at::Tensor& output,
    const at::Tensor& self,
    cnnlTrigonFunctionMode_t mode);

at::Tensor& cnnl_atan2_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& other);

void cnnl_tri_internal(
    at::Tensor& output,
    const at::Tensor& input,
    int64_t diagonal,
    bool tri_up_mode);

void cnnl_lerp_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Tensor& weight);

at::Tensor& cnnl_masked_softmax_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& mask,
    const int axis,
    cnnlMaskedSoftmaxOp_t mode = CNNL_MASKED_SOFTMAX_ADD_MASK);

at::Tensor& cnnl_masked_softmax_dropout_backward_internal(
    at::Tensor& diff_x,
    const at::Tensor& softmax_out,
    const at::Tensor& diff_y,
    const at::Tensor& dropout_dropout_mask,
    const int axis,
    const float p);

at::Tensor& cnnl_conj_internal(at::Tensor& output, const at::Tensor& input);

at::Tensor cnnl_cdist_forward_internal(
    at::Tensor& result,
    const at::Tensor& x1,
    const at::Tensor& x2,
    const double p);

at::Tensor cnnl_cdist_backward_internal(
    at::Tensor& grad_x1,
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& cdist,
    const double p,
    const at::Tensor& grad);

at::Tensor cnnl_repeat_interleave_internal(
    at::Tensor& output,
    const at::Tensor& index,
    const at::Tensor& repeats);

at::Tensor cnnl_linspace_internal(
    at::Tensor& output,
    at::Scalar start,
    at::Scalar end,
    int64_t steps);

at::Tensor& cnnl_det_internal(
    at::Tensor& output,
    const at::Tensor& input,
    std::optional<at::Tensor>& sign_opt, /* for slogdet */
    cnnlDetMode_t mode);

at::Tensor& cnnl_roi_align_internal(
    at::Tensor& output, // output feature map.
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    const double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    const int64_t pooled_height, // The height of the pooled feature map.
    const int64_t pooled_width, // The width of the pooled feature.
    const int64_t sampling_ratio, // The number of points to sample in each bin.
    const bool aligned);

at::Tensor& cnnl_roi_align_backward_internal(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const double spatial_scale,
    const int64_t pooled_height,
    const int64_t pooled_width,
    const int64_t batch_size,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t sampling_ratio,
    const bool aligned,
    at::Tensor& grad_input);

at::Tensor& im2col_out_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& stride);

at::Tensor& col2im_out_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& stride);

at::Tensor cnnl_max_unpool2d_internal(
    at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding);

at::Tensor cnnl_max_unpool2d_backward_internal(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef kernel_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& grad_input);

void cnnl_svd_internal(
    std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>& outputs,
    const at::Tensor& self,
    bool some,
    bool compute_uv,
    const at::Tensor& infos);

std::tuple<at::Tensor, std::optional<at::Tensor>> cnnl_rnnt_loss_internal(
    at::Tensor& logits,
    const at::Tensor& targets,
    const at::Tensor& logit_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    double clamp);

void cnnl_inverse_internal(
    const at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& infos);

void cnnl_bincount_internal(
    const at::Tensor& output,
    const at::Tensor& self,
    const at::Tensor& weight,
    const int64_t min_value,
    const int64_t max_value);

void cnnl_weight_norm_internal(
    at::Tensor& w,
    at::Tensor& norms,
    const at::Tensor& v,
    const at::Tensor& g,
    const int64_t dim);

void cnnl_weight_norm_backward_internal(
    at::Tensor& grad_v,
    at::Tensor& grad_g,
    const at::Tensor& grad_w,
    const at::Tensor& saved_v,
    const at::Tensor& saved_g,
    const at::Tensor& saved_norms,
    const int64_t dim);

at::Tensor& cnnl_polar_internal(
    const at::Tensor& abs,
    const at::Tensor& angle,
    at::Tensor& output);

void cnnl_cross_internal(
    at::Tensor& result,
    const at::Tensor& a,
    const at::Tensor& b,
    const int64_t dim);

void cnnl_trace_internal(const at::Tensor& self, at::Tensor& result);

void cnnl_logaddexp_internal(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& other,
    LogaddexpType base);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cnnl_fa_fwd_internal(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    const bool return_softmax);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cnnl_fa_bwd_internal(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    const bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_mem_eff_fwd_internal(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& bias,
    at::Tensor& output,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const float dropout_p,
    const int64_t custom_mask_type,
    const bool compute_log_sumexp,
    const float scale);

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl_mem_eff_bwd_internal(
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const std::optional<at::Tensor>& bias,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    at::Tensor& db,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    const int64_t custom_mask_type);

template <bool isInplace>
void cnnl_foreach_unary_op(
    at::TensorList tensors,
    at::TensorList outputs,
    const cnnlForeachOpMode_t& mode);

template <typename scalar_t, bool isInplace>
void cnnl_foreach_binary_tensors_op(
    at::TensorList tensors1,
    at::TensorList tensors2,
    at::TensorList outputs,
    const at::ArrayRef<at::Scalar>& scalar_list,
    const scalar_t& scalar,
    const at::Tensor& scalar_tensor,
    const scalar_t& alpha,
    const cnnlForeachOpMode_t& op_mode,
    const cnnlForeachBinaryMode_t& mode);

template <typename scalar_t, bool isInplace>
void cnnl_foreach_lerp_op(
    at::TensorList tensors1,
    at::TensorList tensors2,
    at::TensorList tensors3,
    at::TensorList outputs,
    const at::ArrayRef<at::Scalar>& scalar_list,
    const scalar_t& scalar,
    const cnnlForeachLerpMode_t& mode);

void cnnl_rrelu_with_noise_internal(
    at::Tensor& output,
    const at::Tensor& noise,
    const at::Tensor& self,
    std::optional<at::Generator> gen,
    const float lower,
    const float upper);

std::tuple<at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>>
cnnl_rnn_backward_internal(
    const at::Tensor& input_r,
    const at::TensorList& weight,
    const at::Tensor& weight_buf,
    int64_t weight_stride0,
    const at::Tensor& hx,
    const at::Tensor& cx,
    const at::Tensor& output_r,
    const at::Tensor& grad_output_r,
    const at::Tensor& grad_hy,
    const at::Tensor& grad_cy,
    int64_t fn_mode,
    int64_t fn_hidden_size,
    int64_t proj_size,
    int64_t fn_num_layers,
    double fn_dropout,
    bool fn_train,
    bool fn_bidirectional,
    const int* batch_sizes_int_ptr,
    const at::Tensor& dev_batch_sizes,
    const at::Tensor& fn_dropout_state,
    const at::Tensor& fn_reserve,
    std::array<bool, 3> output_mask);

} // namespace ops
} // namespace torch_mlu
