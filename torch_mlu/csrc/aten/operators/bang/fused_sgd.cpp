#include "aten/operators/bang/bang_kernel.h"
#include "aten/operators/bang/internal/bang_internal.h"
#include "aten/utils/cnnl_util.h"
#include "aten/utils/utils.h"

namespace torch_mlu {
namespace ops {

static bool is_contiguous_tensor(const at::Tensor& t) {
  return t.is_contiguous() || t.is_contiguous(at::MemoryFormat::ChannelsLast) ||
      t.is_contiguous(at::MemoryFormat::ChannelsLast3d);
}

bool bang_fused_sgd(
    const at::Tensor& _dummy_overflow_buf,
    at::TensorList grads,
    at::TensorList params_in,
    at::TensorList momentums,
    const c10::List<std::optional<at::Tensor>>& params_out,
    double weight_decay,
    double momentum,
    double dampening,
    double learning_rate,
    bool nesterov,
    bool first_run,
    bool wd_after_momentum,
    double scale) {
  auto stream = getCurMLUStream();
  auto tensor_num = grads.size();
  cnrtDataType_t in_type =
      cnnlType2CnrtType(getCnnlType(getMluTensorImpl(grads[0])));
  cnrtDataType_t out_type =
      cnnlType2CnrtType(getCnnlType(getMluTensorImpl(params_in[0])));
  int N = params_out.empty() ? 3 : 4;

  cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
  cnrtDim3_t k_dim;
  uint32_t union_number = torch_mlu::getDeviceAttr(cnrtAttrClusterCount);
  uint32_t core_dim = torch_mlu::getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim.x = core_dim;
  k_dim.y = union_number;
  k_dim.z = 1;

  TORCH_CHECK(
      _dummy_overflow_buf.device() == grads[0].device(),
      "expected noop flag to be on the same device as tensors");
  int32_t* overflow = static_cast<int32_t*>(
      mlu_data_ptr(getMluTensorImpl(_dummy_overflow_buf)));

  AddressList g, i, m, o;
  SizeList s;
  int tensor_count = 0;
  std::vector<std::vector<at::Tensor>> contiguous_tensors_list;
  for (int tensor_id = 0; tensor_id < tensor_num; ++tensor_id) {
    at::Tensor grad = grads[tensor_id];
    at::Tensor in = params_in[tensor_id];
    at::Tensor mt = momentums[tensor_id];

    TORCH_CHECK(
        is_contiguous_tensor(grad) && is_contiguous_tensor(in) &&
            is_contiguous_tensor(mt),
        "Found param with index=",
        tensor_id,
        " was not contiguous.");

    int64_t num_elements = grad.numel();
    TORCH_CHECK(
        in.numel() == num_elements,
        "Size mismatch. The numel of grad and param should be equal."
        " but grad is ",
        num_elements,
        ". param is ",
        in.numel());
    TORCH_CHECK(
        mt.numel() == num_elements,
        "Size mismatch. The numel of grad and momentums should be equal."
        " but grad is ",
        num_elements,
        ". momentums is ",
        mt.numel());
    if (num_elements == 0) {
      CNLOG(INFO) << "FusedSGD: Skip zero element tensor.";
      continue;
    }

    std::vector<at::Tensor> contiguous_tensors;
    auto memory_format = in.suggest_memory_format();
    auto grad_contiguous = cnnl_contiguous(grad, memory_format);
    auto in_contiguous = cnnl_contiguous(in, memory_format);
    auto mt_contiguous = cnnl_contiguous(mt, memory_format);

    contiguous_tensors.push_back(in_contiguous);
    contiguous_tensors.push_back(mt_contiguous);

    auto grad_ptr = mlu_data_ptr(getMluTensorImpl(grad_contiguous));
    auto in_ptr = mlu_data_ptr(getMluTensorImpl(in_contiguous));
    auto mt_ptr = mlu_data_ptr(getMluTensorImpl(mt_contiguous));

    g.addresses[tensor_count] = grad_ptr;
    i.addresses[tensor_count] = in_ptr;
    m.addresses[tensor_count] = mt_ptr;
    s.sizes[tensor_count] = num_elements;

    if (N == 4) {
      at::Tensor out = params_out.get(tensor_id).value();
      TORCH_CHECK(
          out.scalar_type() == at::ScalarType::Half,
          "Additional output tensors should always be fp16.");

      auto out_contiguous = cnnl_contiguous(out, memory_format);
      contiguous_tensors.push_back(out_contiguous);
      auto out_ptr = mlu_data_ptr(getMluTensorImpl(out_contiguous));
      o.addresses[tensor_count] = out_ptr;
    }

    contiguous_tensors_list.push_back(contiguous_tensors);

    ++tensor_count;
    if (tensor_count == MAX_TENSOR_NUM) {
      bang_fused_sgd_internal(
          g,
          i,
          m,
          o,
          s,
          tensor_count,
          overflow,
          weight_decay,
          momentum,
          dampening,
          learning_rate,
          nesterov,
          first_run,
          wd_after_momentum,
          scale,
          k_dim,
          k_type,
          stream,
          in_type,
          out_type,
          N);
      tensor_count = 0;
    }
  }

  if (tensor_count != 0) {
    bang_fused_sgd_internal(
        g,
        i,
        m,
        o,
        s,
        tensor_count,
        overflow,
        weight_decay,
        momentum,
        dampening,
        learning_rate,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
        k_dim,
        k_type,
        stream,
        in_type,
        out_type,
        N);
  }

  for (int64_t i = 0; i < tensor_num; ++i) {
    if (is_copy_necessary(params_in[i], contiguous_tensors_list[i][0])) {
      params_in[i].copy_(contiguous_tensors_list[i][0]);
    }
    if (is_copy_necessary(momentums[i], contiguous_tensors_list[i][1])) {
      momentums[i].copy_(contiguous_tensors_list[i][1]);
    }
    if (N == 4) {
      at::Tensor out = params_out.get(i).value();
      if (is_copy_necessary(out, contiguous_tensors_list[i][2])) {
        out.copy_(contiguous_tensors_list[i][2]);
      }
    }
  }

  return true;
}

} // namespace ops
} // namespace torch_mlu
