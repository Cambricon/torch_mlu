#include "aten/operators/cnnl/internal/cnnl_internal.h"

using at::Tensor;
using at::TensorList;

namespace torch_mlu {
namespace ops {

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
    at::IntArrayRef batch_sizes) {
  TORCH_CHECK(
      mode == CNNL_LSTM && cx.defined(),
      "cnnl rnn currently only support LSTM RNN and cx must be defined.");
  TORCH_CHECK(
      num_layers == 1, "cnnl rnn: num_layers does not support greater than 1");
  TORCH_CHECK(
      input.dtype() == at::kDouble || input.dtype() == at::kFloat ||
          input.dtype() == at::kHalf,
      "cnnl rnn: the input data type must be double, float or half");

  auto input_size = batch_sizes.size() != 0 ? input.size(1) : input.size(2);
  auto input_type = getCnnlDataType(input.dtype());
  auto handle = getCurrentHandle();

  // RNNDesc, cnnl lstm not support project.
  CnnlRNNDescriptor rnn_desc;
  rnn_desc.set(
      c10::checked_convert<int32_t, int64_t>(hidden_size, "int32_t"),
      c10::checked_convert<int32_t, int64_t>(proj_size, "int32_t"),
      c10::checked_convert<int32_t, int64_t>(num_layers, "int32_t"),
      c10::checked_convert<int32_t, int64_t>(input_size, "int32_t"),
      has_biases,
      bidirectional,
      mode,
      input_type);

  // Get seq_arr and max_batch_size.
  auto seq_arr_size =
      c10::checked_convert<int32_t, size_t>(batch_sizes.size(), "int32_t");
  int* batch_sizes_int_ptr = nullptr;
  std::vector<int> batch_sizes_int;
  // convert cpu tensor from int64 to int.
  if (seq_arr_size != 0) {
    batch_sizes_int.reserve(seq_arr_size);
    for (size_t i = 0; i < seq_arr_size; ++i) {
      // batch_sizes is store batch size num of packed input,
      batch_sizes_int.push_back(
          c10::checked_convert<int32_t, int64_t>(batch_sizes[i], "int32_t"));
    }
    batch_sizes_int_ptr = batch_sizes_int.data();
  }

  // TensorDesc
  CnnlSeqDataDescriptor input_desc;
  input_desc.set(input, seq_arr_size, batch_sizes_int_ptr);
  auto input_impl = getMluTensorImpl(input);
  auto input_ptr = input_impl->mlu_data_ptr();
  auto hx_impl = getMluTensorImpl(hx);
  auto hx_ptr = hx_impl->mlu_data_ptr();
  auto cx_impl = getMluTensorImpl(cx);
  auto cx_ptr = cx_impl->mlu_data_ptr();
  auto hx_desc = getTensorDesc(hx_impl);
  auto cx_desc = getTensorDesc(cx_impl);
  // dev seq arr space and copy seq arr from cpu to mlu.
  int* dev_seq_lengths_ptr = nullptr;
  at::Tensor dev_seq;
  if (seq_arr_size != 0) {
    const size_t copy_size = seq_arr_size * sizeof(int);
    dev_seq = at::empty(
        {static_cast<long>(seq_arr_size)}, input.options().dtype(at::kInt));
    dev_seq_lengths_ptr = dev_seq.data_ptr<int>();
    auto stream = getCurrentMLUStream();
    CNRT_CHECK(cnrtMemcpyAsync_V2(
        (void*)dev_seq_lengths_ptr,
        (void*)batch_sizes_int_ptr,
        copy_size,
        stream.stream(),
        cnrtMemcpyHostToDev));
    stream.synchronize();
  }

  // RNNTempSize
  size_t reserve_size = 0;
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetRNNTempSizes(
      handle,
      rnn_desc.desc(),
      input_desc.desc(),
      &workspace_size,
      &reserve_size));

  // RNNWorkspace
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  auto reserve = at::empty(
      {static_cast<long>(reserve_size)}, input.options().dtype(at::kByte));
  auto reserve_impl = getMluTensorImpl(reserve);
  auto reserve_ptr = reserve_impl->mlu_data_ptr();

  // Output
  auto hy = at::empty(hx.sizes(), hx.options());
  auto hy_impl = getMluTensorImpl(hy);
  auto hy_ptr = hy_impl->mlu_data_ptr();
  auto cy = at::empty(cx.sizes(), cx.options());
  auto cy_impl = getMluTensorImpl(cy);
  auto cy_ptr = cy_impl->mlu_data_ptr();

  int64_t num_directions = bidirectional ? 2 : 1;
  std::vector<int64_t> output_size = input.sizes().vec();
  output_size.back() =
      num_directions * (proj_size > 0 ? proj_size : hidden_size);
  auto output = at::empty(output_size, input.options());
  auto output_impl = getMluTensorImpl(output);
  auto output_ptr = output_impl->mlu_data_ptr();

  CnnlSeqDataDescriptor output_desc;
  output_desc.set(output, seq_arr_size, batch_sizes_int_ptr);

  // WeightBuf
  std::vector<at::Tensor> vec;
  if (has_biases && proj_size > 0) {
    // Different with cudnn which use cudnnGetRNNLinLayerMatrixParams to calc
    // the position of w_hr, cnnl use the fixed order to try to align with
    // cudnn, w_hr is infront of bias.
    for (auto& i : {0, 1, 4, 2, 3}) {
      vec.emplace_back(params[i].reshape({-1}));
    }
    if (bidirectional) {
      for (auto& i : {5, 6, 9, 7, 8}) {
        vec.emplace_back(params[i].reshape({-1}));
      }
    }
  } else {
    for (auto& p : params) {
      vec.emplace_back(p.reshape({-1}));
    }
  }
  auto weight = at::cat(TensorList(vec));
  auto weight_impl = getMluTensorImpl(weight);
  auto weight_ptr = weight_impl->mlu_data_ptr();
  // onchip dtype is float when the dtype of tensor is double.
  auto weight_nbytes =
      weight.dtype() == at::kDouble ? weight.nbytes() / 2 : weight.nbytes();

  // TODO(PYTORCH-9423): May be optimized by cnnlRNNForwardInference.
  // ForwardTraining
  TORCH_CNNL_CHECK(cnnlRNNForwardTraining(
      handle,
      rnn_desc.desc(),
      dev_seq_lengths_ptr,
      input_desc.desc(),
      input_ptr,
      output_desc.desc(),
      output_ptr,
      hx_desc.get(),
      hx_ptr,
      hy_ptr,
      cx_desc.get(),
      cx_ptr,
      cy_ptr,
      weight_ptr,
      weight_nbytes,
      workspace_ptr.get(),
      workspace_size,
      reserve_ptr,
      reserve_size));

  return std::make_tuple(output, hy, cy, reserve, weight);
}

} // namespace ops
} // namespace torch_mlu
