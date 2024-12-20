#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"
#include "aten/utils/utils.h"

namespace torch_mlu {
namespace ops {

bool bang_fused_lamb(
    const at::Tensor& _dummy_overflow_buf,
    at::TensorList grads,
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    int64_t step,
    int64_t bias_correction,
    double weight_decay,
    int64_t grad_averaging,
    int64_t mode,
    const at::Tensor& global_grad_norm,
    double max_grad_norm,
    bool use_nvlamb_python) {
  auto stream = getCurMLUStream();
  auto tensor_num = grads.size();
  cnrtDataType_V2_t cnrt_type =
      cnnlType2CnrtType_V2(getCnnlType(getMluTensorImpl(grads[0])));

  cnrtFunctionType_t k_type = cnrtFuncTypeUnion1;
  cnrtDim3_t k_dim;
  uint32_t union_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.x = core_dim;
  k_dim.y = union_number;
  k_dim.z = 1;

  TORCH_CHECK(
      global_grad_norm.scalar_type() == at::ScalarType::Float,
      "MultiTensorLAMB: The data type of global_grad_norm must be float");
  TORCH_CHECK(
      _dummy_overflow_buf.scalar_type() == at::ScalarType::Int,
      "MultiTensorLAMB: The data type of overflow must be int32")

  float* global_grad_norm_ptr =
      static_cast<float*>(mlu_data_ptr(getMluTensorImpl(global_grad_norm)));
  int32_t* overflow = static_cast<int32_t*>(
      mlu_data_ptr(getMluTensorImpl(_dummy_overflow_buf)));

  // compute workspace
  auto workspace_size = sizeof(float) * (tensor_num * 2 + 1);
  at::Tensor workspace = at::zeros(
      workspace_size,
      at::TensorOptions().dtype(at::kChar).device(at::kPrivateUse1));
  float* fp_workspace =
      static_cast<float*>(mlu_data_ptr(getMluTensorImpl(workspace)));

  float beta1_correction_recip = 1;
  float beta2_correction_recip = 1;
  if (bias_correction == 1) {
    beta1_correction_recip = 1 / (1 - std::pow(beta1, step));
    beta2_correction_recip = 1 / (1 - std::pow(beta2, step));
  }
  float epsilon_correction = epsilon / std::sqrt(beta2_correction_recip);
  float correction_rate =
      std::sqrt(beta2_correction_recip) / beta1_correction_recip;

  AddressList g, m, v, p;
  SizeList sizes;
  int tensor_count = 0;
  int tensor_offset = 0;
  std::vector<std::vector<at::Tensor>> contiguous_tensors_list;
  for (int tensor_id = 0; tensor_id < tensor_num; ++tensor_id) {
    at::Tensor grad = grads[tensor_id];
    at::Tensor exp_avg = exp_avgs[tensor_id];
    at::Tensor exp_avg_sq = exp_avg_sqs[tensor_id];
    at::Tensor param = params[tensor_id];
    int64_t num_elements = grad.numel();
    std::vector<at::Tensor> contiguous_tensors;

    auto memory_format = param.suggest_memory_format();
    auto grad_contiguous = cnnl_contiguous(grad, memory_format);
    auto exp_avg_contiguous = cnnl_contiguous(exp_avg, memory_format);
    auto exp_avg_sq_contiguous = cnnl_contiguous(exp_avg_sq, memory_format);
    auto param_contiguous = cnnl_contiguous(param, memory_format);

    contiguous_tensors.push_back(exp_avg_contiguous);
    contiguous_tensors.push_back(exp_avg_sq_contiguous);
    contiguous_tensors.push_back(param_contiguous);
    contiguous_tensors_list.push_back(contiguous_tensors);

    if (num_elements == 0) {
      CNLOG(INFO) << "MultiTensorLAMB: Skip zero element tensor.";
      continue;
    }

    g.addresses[tensor_count] = mlu_data_ptr(getMluTensorImpl(grad_contiguous));
    m.addresses[tensor_count] =
        mlu_data_ptr(getMluTensorImpl(exp_avg_contiguous));
    v.addresses[tensor_count] =
        mlu_data_ptr(getMluTensorImpl(exp_avg_sq_contiguous));
    p.addresses[tensor_count] =
        mlu_data_ptr(getMluTensorImpl(param_contiguous));
    sizes.sizes[tensor_count] = num_elements;

    ++tensor_count;
    if (tensor_count == MAX_TENSOR_NUM) {
      bang_fused_lamb_internal(
          g,
          p,
          m,
          v,
          sizes,
          global_grad_norm_ptr,
          tensor_count,
          learning_rate,
          beta1,
          beta2,
          epsilon_correction,
          weight_decay,
          correction_rate,
          max_grad_norm,
          mode,
          grad_averaging,
          fp_workspace + tensor_offset,
          fp_workspace + tensor_offset + tensor_num,
          overflow,
          use_nvlamb_python,
          k_dim,
          k_type,
          stream,
          cnrt_type);
      tensor_count = 0;
      tensor_offset = tensor_id + 1;
    }
  }

  if (tensor_count != 0) {
    bang_fused_lamb_internal(
        g,
        p,
        m,
        v,
        sizes,
        global_grad_norm_ptr,
        tensor_count,
        learning_rate,
        beta1,
        beta2,
        epsilon_correction,
        weight_decay,
        correction_rate,
        max_grad_norm,
        mode,
        grad_averaging,
        fp_workspace + tensor_offset,
        fp_workspace + tensor_offset + tensor_num,
        overflow,
        use_nvlamb_python,
        k_dim,
        k_type,
        stream,
        cnrt_type);
  }

  for (int64_t i = 0; i < tensor_num; ++i) {
    if (is_copy_necessary(exp_avgs[i], contiguous_tensors_list[i][0])) {
      exp_avgs[i].copy_(contiguous_tensors_list[i][0]);
    }
    if (is_copy_necessary(exp_avg_sqs[i], contiguous_tensors_list[i][1])) {
      exp_avg_sqs[i].copy_(contiguous_tensors_list[i][1]);
    }
    if (is_copy_necessary(params[i], contiguous_tensors_list[i][2])) {
      params[i].copy_(contiguous_tensors_list[i][2]);
    }
  }

  return true;
}

} // namespace ops
} // namespace torch_mlu
