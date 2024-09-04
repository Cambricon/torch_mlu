#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"
#include "aten/utils/utils.h"

namespace torch_mlu {
namespace ops {

void _fused_lamb_amp_common(
    const at::Tensor& noop_flag,
    std::vector<std::vector<at::Tensor>>& contiguous_tensors_list,
    at::TensorList grads,
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    const at::Tensor& learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    const at::Tensor& step,
    int64_t bias_correction,
    double weight_decay,
    int64_t grad_averaging,
    int64_t mode,
    const at::Tensor& global_grad_norm,
    const at::Tensor& max_grad_norm,
    bool use_nvlamb_python,
    const at::Tensor& inv_scale) {
  TORCH_CHECK(
      global_grad_norm.scalar_type() == at::ScalarType::Float,
      "MultiTensorLAMBAMP: The data type of global_grad_norm must be float");
  TORCH_CHECK(
      learning_rate.scalar_type() == at::ScalarType::Float,
      "MultiTensorLAMBAMP: The data type of learning_rate must be float");
  TORCH_CHECK(
      step.scalar_type() == at::ScalarType::Int,
      "MultiTensorLAMBAMP: The data type of step must be int32")
  TORCH_CHECK(
      noop_flag.scalar_type() == at::ScalarType::Int,
      "MultiTensorLAMBAMP: The data type of overflow must be int32")
  TORCH_CHECK(
      exp_avgs[0].scalar_type() == at::ScalarType::Float,
      "MultiTensorLAMBAMP: The data type of exp_avgs must be float");
  TORCH_CHECK(
      exp_avg_sqs[0].scalar_type() == at::ScalarType::Float,
      "MultiTensorLAMBAMP: The data type of exp_avg_sqs must be float");

  auto beta1_cvt = c10::checked_convert<float, double>(beta1, "float");
  auto beta2_cvt = c10::checked_convert<float, double>(beta2, "float");
  auto epsilon_cvt = c10::checked_convert<float, double>(epsilon, "float");
  auto bias_correction_cvt =
      c10::checked_convert<int, int64_t>(bias_correction, "int");
  auto weight_decay_cvt =
      c10::checked_convert<float, double>(weight_decay, "float");
  auto grad_averaging_cvt =
      c10::checked_convert<int, int64_t>(grad_averaging, "int");
  auto mode_cvt = c10::checked_convert<int, int64_t>(mode, "int");

  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  uint32_t union_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.x = core_dim;
  k_dim.y = union_number;
  k_dim.z = 1;

  int* noop_flag_ptr =
      static_cast<int*>(getMluTensorImpl(noop_flag)->mlu_data_ptr());
  float* lr_ptr =
      static_cast<float*>(getMluTensorImpl(learning_rate)->mlu_data_ptr());
  int* step_ptr = static_cast<int*>(getMluTensorImpl(step)->mlu_data_ptr());
  float* ggn_ptr =
      static_cast<float*>(getMluTensorImpl(global_grad_norm)->mlu_data_ptr());
  float* inv_scale_ptr =
      static_cast<float*>(getMluTensorImpl(inv_scale)->mlu_data_ptr());
  float* max_grad_norm_ptr =
      static_cast<float*>(getMluTensorImpl(max_grad_norm)->mlu_data_ptr());

  // compute workspace
  auto tensor_num = grads.size();
  auto workspace_size = sizeof(float) * (tensor_num * 2 + 1);
  at::Tensor workspace = at::zeros(
      workspace_size,
      at::TensorOptions().dtype(at::kChar).device(at::kPrivateUse1));
  float* fp_workspace =
      static_cast<float*>(getMluTensorImpl(workspace)->mlu_data_ptr());

  cnrtDataType_t grad_cnrt_type =
      cnnlType2CnrtType(getCnnlType(getMluTensorImpl(grads[0])));
  cnrtDataType_t param_cnrt_type =
      cnnlType2CnrtType(getCnnlType(getMluTensorImpl(params[0])));
  auto stream = getCurMLUStream();
  AddressList g, m, v, p, half_p;
  SizeList sizes;
  int tensor_count = 0, tensor_offset = 0;
  const bool is_amp = !contiguous_tensors_list[0].empty();
  for (size_t tensor_id = 0; tensor_id < tensor_num; ++tensor_id) {
    at::Tensor grad = grads[tensor_id];
    at::Tensor exp_avg = exp_avgs[tensor_id];
    at::Tensor exp_avg_sq = exp_avg_sqs[tensor_id];
    at::Tensor param = params[tensor_id];
    auto num_elements = grad.numel();

    auto memory_format = param.suggest_memory_format();
    auto grad_contiguous = cnnl_contiguous(grad, memory_format);
    auto exp_avg_contiguous = cnnl_contiguous(exp_avg, memory_format);
    auto exp_avg_sq_contiguous = cnnl_contiguous(exp_avg_sq, memory_format);
    auto param_contiguous = cnnl_contiguous(param, memory_format);
    contiguous_tensors_list[tensor_id].push_back(exp_avg_contiguous);
    contiguous_tensors_list[tensor_id].push_back(exp_avg_sq_contiguous);
    contiguous_tensors_list[tensor_id].push_back(param_contiguous);

    if (num_elements == 0) {
      CNLOG(INFO) << "MultiTensorLAMB: Skip zero element tensor.";
      continue;
    }

    g.addresses[tensor_count] =
        getMluTensorImpl(grad_contiguous)->mlu_data_ptr();
    m.addresses[tensor_count] =
        getMluTensorImpl(exp_avg_contiguous)->mlu_data_ptr();
    v.addresses[tensor_count] =
        getMluTensorImpl(exp_avg_sq_contiguous)->mlu_data_ptr();
    p.addresses[tensor_count] =
        getMluTensorImpl(param_contiguous)->mlu_data_ptr();
    if (is_amp) {
      auto dst_param_contiguous = contiguous_tensors_list[tensor_id][0];
      half_p.addresses[tensor_count] =
          getMluTensorImpl(dst_param_contiguous)->mlu_data_ptr();
    }
    sizes.sizes[tensor_count] = num_elements;

    ++tensor_count;
    if (tensor_count == MAX_TENSOR_NUM) {
      bang_fused_lamb_amp_internal(
          g,
          p,
          m,
          v,
          half_p,
          sizes,
          tensor_count,
          noop_flag_ptr,
          beta1_cvt,
          beta2_cvt,
          grad_averaging_cvt,
          step_ptr,
          bias_correction_cvt,
          epsilon_cvt,
          mode_cvt,
          weight_decay_cvt,
          ggn_ptr,
          max_grad_norm_ptr,
          fp_workspace + tensor_offset,
          fp_workspace + tensor_offset + tensor_num,
          inv_scale_ptr,
          lr_ptr,
          use_nvlamb_python,
          k_dim,
          k_type,
          stream,
          grad_cnrt_type,
          param_cnrt_type);
      tensor_count = 0;
      tensor_offset = tensor_id + 1;
    }
  }

  if (tensor_count != 0) {
    bang_fused_lamb_amp_internal(
        g,
        p,
        m,
        v,
        half_p,
        sizes,
        tensor_count,
        noop_flag_ptr,
        beta1_cvt,
        beta2_cvt,
        grad_averaging_cvt,
        step_ptr,
        bias_correction_cvt,
        epsilon_cvt,
        mode_cvt,
        weight_decay_cvt,
        ggn_ptr,
        max_grad_norm_ptr,
        fp_workspace + tensor_offset,
        fp_workspace + tensor_offset + tensor_num,
        inv_scale_ptr,
        lr_ptr,
        use_nvlamb_python,
        k_dim,
        k_type,
        stream,
        grad_cnrt_type,
        param_cnrt_type);
  }

  const auto n_tensors_list = contiguous_tensors_list[0].size();
  for (size_t i = 0; i < tensor_num; ++i) {
    if (is_copy_necessary(
            exp_avgs[i], contiguous_tensors_list[i][n_tensors_list - 3])) {
      exp_avgs[i].copy_(contiguous_tensors_list[i][n_tensors_list - 3]);
    }
    if (is_copy_necessary(
            exp_avg_sqs[i], contiguous_tensors_list[i][n_tensors_list - 2])) {
      exp_avg_sqs[i].copy_(contiguous_tensors_list[i][n_tensors_list - 2]);
    }
    if (is_copy_necessary(
            params[i], contiguous_tensors_list[i][n_tensors_list - 1])) {
      params[i].copy_(contiguous_tensors_list[i][n_tensors_list - 1]);
    }
  }
}

bool bang_fused_lamb_amp(
    const at::Tensor& noop_flag,
    at::TensorList grads,
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    const at::Tensor& learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    const at::Tensor& step,
    int64_t bias_correction,
    double weight_decay,
    int64_t grad_averaging,
    int64_t mode,
    const at::Tensor& global_grad_norm,
    const at::Tensor& max_grad_norm,
    bool use_nvlamb_python,
    const at::Tensor& inv_scale) {
  if (grads.size() == 0)
    return true;

  TORCH_CHECK(
      grads[0].scalar_type() == params[0].scalar_type(),
      "MultiTensorLAMBAMP: The data type of grads and params must be the same")

  std::vector<at::Tensor> sub_tensors_list;
  std::vector<std::vector<at::Tensor>> contiguous_tensors_list(
      grads.size(), sub_tensors_list);
  _fused_lamb_amp_common(
      noop_flag,
      contiguous_tensors_list,
      grads,
      params,
      exp_avgs,
      exp_avg_sqs,
      learning_rate,
      beta1,
      beta2,
      epsilon,
      step,
      bias_correction,
      weight_decay,
      grad_averaging,
      mode,
      global_grad_norm,
      max_grad_norm,
      use_nvlamb_python,
      inv_scale);

  return true;
}

bool bang_fused_lamb_amp(
    const at::Tensor& noop_flag,
    at::TensorList grads,
    at::TensorList params,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList dst_param_fp16s,
    const at::Tensor& learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    const at::Tensor& step,
    int64_t bias_correction,
    double weight_decay,
    int64_t grad_averaging,
    int64_t mode,
    const at::Tensor& global_grad_norm,
    const at::Tensor& max_grad_norm,
    bool use_nvlamb_python,
    const at::Tensor& inv_scale) {
  if (grads.size() == 0)
    return true;

  TORCH_CHECK(
      grads[0].scalar_type() == at::ScalarType::Half,
      "MultiTensorLAMBAMP: The data type of grad should be half.")
  TORCH_CHECK(
      dst_param_fp16s[0].scalar_type() == at::ScalarType::Half,
      "MultiTensorLAMBAMP: The data type of dst_param_fp16s should be half.")
  TORCH_CHECK(
      params[0].scalar_type() == at::ScalarType::Float,
      "MultiTensorLAMBAMP: The data type of param should be float.")

  std::vector<at::Tensor> sub_tensors_list;
  std::vector<std::vector<at::Tensor>> contiguous_tensors_list(
      grads.size(), sub_tensors_list);
  const auto tensor_num = grads.size();
  for (size_t tensor_id = 0; tensor_id < tensor_num; ++tensor_id) {
    at::Tensor param = params[tensor_id];
    at::Tensor dst_param = dst_param_fp16s[tensor_id];
    auto memory_format = param.suggest_memory_format();
    auto dst_param_contiguous = cnnl_contiguous(dst_param, memory_format);
    contiguous_tensors_list[tensor_id].push_back(dst_param_contiguous);
  }

  _fused_lamb_amp_common(
      noop_flag,
      contiguous_tensors_list,
      grads,
      params,
      exp_avgs,
      exp_avg_sqs,
      learning_rate,
      beta1,
      beta2,
      epsilon,
      step,
      bias_correction,
      weight_decay,
      grad_averaging,
      mode,
      global_grad_norm,
      max_grad_norm,
      use_nvlamb_python,
      inv_scale);

  for (size_t i = 0; i < tensor_num; ++i) {
    if (is_copy_necessary(dst_param_fp16s[i], contiguous_tensors_list[i][0]))
      dst_param_fp16s[i].copy_(contiguous_tensors_list[i][0]);
  }

  return true;
}

} // namespace ops
} // namespace torch_mlu
