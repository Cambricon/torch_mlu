/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "aten/cnnl/cnnlOpDescriptors.h"

namespace torch_mlu {
void CnnlPoolingDescriptor::set(
    cnnlPoolingMode_t mode,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_u,
    int pad_d,
    int pad_l,
    int pad_r,
    bool ceil_mode) {
  TORCH_CNNL_CHECK(cnnlSetPooling2dDescriptor_v2(
      this->mut_desc(),
      mode,
      CNNL_PROPAGATE_NAN,
      kernel_h,
      kernel_w,
      pad_u,
      pad_d,
      pad_l,
      pad_r,
      stride_h,
      stride_w,
      1,
      1,
      ceil_mode));
}

void CnnlPoolingDescriptor::set(
    cnnlPoolingMode_t mode,
    int64_t dims,
    const int kernel_size[],
    const int stride[],
    const int padding[],
    const int dilation[],
    bool ceil_mode) {
  TORCH_CNNL_CHECK(cnnlSetPoolingNdDescriptor_v2(
      this->mut_desc(),
      mode,
      CNNL_NOT_PROPAGATE_NAN,
      dims,
      kernel_size,
      padding,
      stride,
      dilation,
      ceil_mode));
}

void CnnlTransposeDescriptor::set(const int p_dims, const int permute[]) {
  TORCH_CNNL_CHECK(
      cnnlSetTransposeDescriptor(this->mut_desc(), p_dims, permute));
}

void CnnlReduceDescriptor::set(
    std::vector<int64_t> axis,
    cnnlReduceOp_t mode,
    cnnlReduceIndices_t is_indices,
    cnnlIndicesType_t indices_type,
    cnnlDataType_t tensor_type,
    float p) {
  int axis_num = axis.size();
  std::vector<int> axis_list(axis_num);
  for (int i = 0; i < axis_num; i++) {
    axis_list[i] = static_cast<int>(axis[i]);
  }
  TORCH_CNNL_CHECK(cnnlSetReduceDescriptor_v2(
      this->mut_desc(),
      axis_list.data(),
      axis_num,
      mode,
      tensor_type,
      CNNL_NOT_PROPAGATE_NAN,
      is_indices,
      indices_type,
      p));
}

void CnnlStdVarMeanDescriptor::set(
    std::vector<int> axis,
    cnnlStdVarMeanOp_t mode,
    bool unbiased) {
  TORCH_CNNL_CHECK(cnnlSetStdVarMeanDescriptor(
      this->mut_desc(), mode, axis.size(), axis.data(), unbiased));
}

void CnnlOpTensorDescriptor::set(
    cnnlOpTensorDesc_t op_type,
    cnnlDataType_t op_tensor_comp_type,
    cnnlNanPropagation_t op_tensor_nan_opt) {
  TORCH_CNNL_CHECK(cnnlSetOpTensorDescriptor(
      this->mut_desc(), op_type, op_tensor_comp_type, op_tensor_nan_opt));
}

// For cnnlSetActivationDescriptor_v6 API, ceof, gamma, scale
// only support float type.
void CnnlActivationDescriptor::set(
    cnnlActivationMode_t mode,
    cnnlActivationPreference_t prefer,
    float ceof,
    int64_t sliced_dim,
    float gamma,
    float scale,
    bool is_result,
    bool approximate) {
  TORCH_CNNL_CHECK(cnnlSetActivationDescriptor_v6(
      this->mut_desc(),
      mode,
      prefer,
      CNNL_PROPAGATE_NAN,
      ceof,
      static_cast<int>(sliced_dim),
      gamma,
      scale,
      is_result,
      approximate));
}

void CnnlGridSampleDescriptor::set(
    const cnnlInterpMode_t interp_mode,
    const cnnlGridSamplePaddingMode_t padding_mode,
    const bool align_corners) {
  TORCH_CNNL_CHECK(cnnlSetGridSampleDescriptor(
      this->mut_desc(), interp_mode, padding_mode, align_corners));
}

void CnnlUniqueDescriptor::set(
    bool sorted,
    int dim,
    bool return_inverse,
    bool return_counts) {
  TORCH_CNNL_CHECK(cnnlSetUniqueDescriptor(
      this->mut_desc(),
      sorted ? CNNL_SORT_ASCEND : CNNL_UNSORT_REVERSE,
      dim,
      return_inverse,
      return_counts));
}

void CnnlUniqueConsecutiveDescriptor::set(
    int dim,
    bool return_inverse,
    bool return_counts) {
  TORCH_CNNL_CHECK(cnnlSetUniqueConsecutiveDescriptor(
      this->mut_desc(), dim, return_inverse, return_counts));
}

void CnnlMatmulDescriptor::set_attr(
    cnnlMatMulDescAttribute_t attr,
    const void* buf,
    size_t size_in_bytes) {
  TORCH_CNNL_CHECK(
      cnnlSetMatMulDescAttr(this->mut_desc(), attr, buf, size_in_bytes));
}

void CnnlMatmulExDescriptor::set_attr(
    cnnlMatMulExDescAttribute_t attr,
    const void* buf,
    size_t size_in_bytes) {
  TORCH_CNNL_CHECK(
      cnnlSetMatMulExDescAttr(this->mut_desc(), attr, buf, size_in_bytes));
}

void CnnlStrideBatchMatmulDescriptor::set_attr(
    cnnlStrideBatchMatMulDescAttribute_t attr,
    const void* buf,
    size_t size_in_bytes) {
  TORCH_CNNL_CHECK(cnnlSetStrideBatchMatMulDescAttr(
      this->mut_desc(), attr, buf, size_in_bytes));
}

void CnnlCTCLossDescriptor::set(
    cnnlCTCLossNormalizationMode_t norm_mode,
    cnnlCTCLossReduceMode_t reduce_mode,
    cnnlCTCLossZeroInfinityMode_t zero_infinity,
    int blank,
    int max_input_length,
    int max_label_length) {
  TORCH_CNNL_CHECK(cnnlSetCTCLossDescriptor(
      this->mut_desc(),
      norm_mode,
      reduce_mode,
      zero_infinity,
      blank,
      max_input_length,
      max_label_length));
}

void CnnlNmsDescriptor::set(
    const cnnlNmsBoxPointMode_t box_mode,
    const cnnlNmsOutputMode_t output_mode,
    const cnnlNmsAlgo_t algo,
    const cnnlNmsMethodMode_t method_mode,
    const float iou_threshold,
    const float soft_nms_sigma,
    const int max_output_size,
    const float confidence_threshold,
    const float offset,
    const int input_layout,
    const bool pad_to_max_output_size) {
  TORCH_CNNL_CHECK(cnnlSetNmsDescriptor_v5(
      this->mut_desc(),
      box_mode,
      output_mode,
      algo,
      method_mode,
      iou_threshold,
      soft_nms_sigma,
      max_output_size,
      confidence_threshold,
      offset,
      input_layout,
      pad_to_max_output_size));
}

void CnnlRoiAlignDescriptor::set(
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float spatial_scale,
    const int pool_mode,
    const bool aligned) {
  TORCH_CNNL_CHECK(cnnlSetRoiAlignDescriptor_v2(
      this->mut_desc(),
      pooled_height,
      pooled_width,
      sampling_ratio,
      spatial_scale,
      pool_mode,
      aligned));
}
void CnnlDCNDescriptor::set(
    int dim,
    int* padding,
    int* stride,
    int* dilation,
    int64_t deformable_group,
    int64_t conv_group,
    int64_t im2col_step,
    cnnlDataType_t dtype) {
  TORCH_CHECK(dim == 4, "DCN input's dim must equal to 4.");
  int n = dim - 2;
  int p_dim = 4;
  std::vector<int> padding_vec(p_dim, 0);
  std::vector<int> stride_vec(n, 0);
  std::vector<int> dilation_vec(n, 0);
  int deformable_group_t;
  int conv_group_t;
  int im2col_step_t;
  for (int i = 0; i < p_dim; i++) {
    padding_vec[i] = padding[i];
  }
  for (int i = 0; i < n; i++) {
    stride_vec[i] = stride[i];
    dilation_vec[i] = dilation[i];
  }
  deformable_group_t = deformable_group;
  conv_group_t = conv_group;
  im2col_step_t = im2col_step;
  TORCH_CNNL_CHECK(cnnlSetDCNDescriptor(
      mut_desc(),
      dim,
      padding_vec.data(),
      stride_vec.data(),
      dilation_vec.data(),
      deformable_group_t,
      conv_group_t,
      im2col_step_t,
      dtype));
}

void CnnlRNNDescriptor::set(
    int hidden_size,
    int proj_size,
    int num_layers,
    int input_size,
    bool has_biases,
    bool bidirectional,
    cnnlRNNMode_t mode,
    cnnlDataType_t input_type) {
  auto math_type = input_type;
  // TODO(PYTORCH-9425): Prompote dtype like cudnn.
  // if (math_type == CNNL_DTYPE_HALF)
  //   math_type = CNNL_DTYPE_FLOAT;
  auto input_mode = CNNL_RNN_LINEAR_INPUT;
  auto bidirectional_t =
      bidirectional ? CNNL_RNN_BIDIRECTIONAL : CNNL_RNN_UNIDIRECTIONAL;
  auto bias_mode = has_biases ? CNNL_RNN_DOUBLE_BIAS : CNNL_RNN_NO_BIAS;

  TORCH_CNNL_CHECK(cnnlSetRNNDescriptor_v2(
      mut_desc(),
      mode,
      bias_mode,
      bidirectional_t,
      input_mode,
      input_type,
      math_type,
      input_size,
      hidden_size,
      proj_size,
      num_layers,
      nullptr,
      CNNL_RNN_PADDED_IO_DISABLED));
}

// For LSTM packed or padded mode, cnnl need tensor dims equal to 3.
void CnnlSeqDataDescriptor::set(
    const at::Tensor& t,
    int seq_array,
    const int* seq_array_ptr) {
  // Only support 3 dims. packed input dim == 2; padded input dim == 3.
  int t_dim = 3;
  const bool is_input_packed = seq_array != 0 ? true : false;
  auto layout = is_input_packed ? CNNL_SEQDATA_TNC_PACKED : CNNL_SEQDATA_TNC;
  auto* tensor_impl = getMluTensorImpl(t);
  cnnlDataType_t dtype = getCnnlType(tensor_impl);
  std::vector<int64_t> cnnl_size;
  cnnl_size.reserve(t_dim);
  cnnl_size[0] = is_input_packed ? seq_array : t.size(0);
  cnnl_size[1] = is_input_packed ? seq_array_ptr[0] : t.size(1);
  cnnl_size[2] = is_input_packed ? t.size(1) : t.size(2);
  TORCH_CNNL_CHECK(cnnlSetSeqDataDescriptor_v2(
      mut_desc(),
      layout,
      dtype,
      t_dim,
      cnnl_size.data(),
      seq_array,
      seq_array_ptr,
      nullptr));
}

void CnnlTrigonDescriptor::set(
    cnnlTrigonFunctionMode_t mode,
    cnnlComputationPreference_t prefer) {
  TORCH_CNNL_CHECK(cnnlSetTrigonDescriptor_v2(this->mut_desc(), mode, prefer));
}

void CnnlInterpDescriptor::set(
    cnnlTensorDescriptor_t input_desc,
    cnnlInterpMode_t mode,
    bool align_corners,
    bool align_center,
    float* scales,
    bool is_exact) {
  cnnlInterpCoordinateTransformationMode_t coordinate_trans_mode;
  // only used for nearest
  cnnlInterpRoundMode_t nearest_round_mode;
  if (!align_corners && align_center) {
    coordinate_trans_mode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO0;
    nearest_round_mode = CNNL_INTERP_CEIL;
  } else if (align_corners && !align_center) {
    coordinate_trans_mode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO2;
    nearest_round_mode = CNNL_INTERP_ROUND_PERFER_CEIL;
  } else if (!align_corners && !align_center) {
    coordinate_trans_mode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO3;
    nearest_round_mode = CNNL_INTERP_FLOOR;
  } else {
    TORCH_CHECK(
        false, "unsupported combination of align_corners and align_centers");
  }
  // especially, TRANSFORMATION_ALGO7 for 'nearest-exact'
  if (is_exact) {
    coordinate_trans_mode = CNNL_INTERP_COORDINATE_TRANSFORMATION_ALGO7;
  }
  TORCH_CNNL_CHECK(
      cnnlSetInterpDescriptor(this->mut_desc(), mode, coordinate_trans_mode));
  TORCH_CNNL_CHECK(cnnlSetInterpDescriptorEx(
      this->mut_desc(),
      input_desc,
      nearest_round_mode,
      scales,
      nullptr,
      -0.75,
      false));
}

void CnnlInterpBackwardDescriptor::set(
    cnnlTensorDescriptor_t input_desc,
    cnnlInterpMode_t mode,
    cnnlInterpAlgo_t algo,
    const double* scales) {
  TORCH_CNNL_CHECK(cnnlSetInterpDescriptor_v2(
      this->mut_desc(), input_desc, mode, algo, scales));
}

void CnnlHistogramDescriptor::set(
    const cnnlHistogramMode_t hist_mode,
    int bins,
    float range_min,
    float range_max,
    bool ext_range,
    bool is_index) {
  if (hist_mode == CNNL_HISTOGRAM_MODE_HISTO_COUNT) {
    TORCH_CNNL_CHECK(cnnlSetHistogramDescriptorHistoCountMode(
        this->mut_desc(), bins, range_min, range_max, ext_range, is_index));
  } else {
    AT_ERROR("currently cnnlhistogram only support histc mode!");
  }
}

void CnnlHistogramDescriptor::set(
    const int64_t value_min,
    const int64_t value_max,
    const int64_t axis,
    bool binary_out) {
  TORCH_CNNL_CHECK(cnnlSetHistogramDescriptorBinCountMode(
      this->mut_desc(), value_min, value_max, axis, binary_out));
}

void CnnlRNNTLossDescriptor::set(
    const cnnlLossReduction_t reduce_mode,
    const int64_t blank,
    const double clamp,
    const bool fused_log_softmax,
    const int64_t max_logit_length,
    const int64_t max_target_length) {
  TORCH_CNNL_CHECK(cnnlSetRNNTLossDescriptor(
      this->mut_desc(),
      reduce_mode,
      blank,
      clamp,
      fused_log_softmax,
      max_logit_length,
      max_target_length));
}

void CnnlEmbeddingBagDescriptor::set(
    const cnnlReduceMode_t mode,
    const void* max_norm,
    const void* norm_type,
    const void* padding_idx,
    const bool scale_grad_by_freq,
    const bool include_last_offset) {
  TORCH_CNNL_CHECK(cnnlSetEmbeddingBagDescriptor(
      this->mut_desc(),
      mode,
      max_norm,
      norm_type,
      padding_idx,
      scale_grad_by_freq,
      include_last_offset));
}
} // end of namespace torch_mlu
