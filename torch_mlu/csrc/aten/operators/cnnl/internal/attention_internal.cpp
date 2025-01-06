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
#include "cnnl_extra.h"
#include "framework/generator/generator_impl.h"
#include "aten/operators/cnnl/internal/philox_utils.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

#define CHECK_SHAPE(x, ...)                        \
  TORCH_CHECK(                                     \
      x.sizes() == at::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")

namespace {

inline int64_t getCounterOffset(size_t nelem, int64_t thread_num) {
  const int64_t UNROLL = 4;
  return ((nelem - 1) / (thread_num * UNROLL) + 1) * UNROLL;
}

} // anonymous namespace

struct MaskParamsFwd {
  bool is_causal;
  int left_size = -1;
  int right_size = -1;
  cnnlAttentionMaskMode_t attn_mask_mode;
  MaskParamsFwd(
      const int64_t ori_left_size,
      const int64_t ori_right_size,
      const int64_t max_seq_q,
      const int64_t max_seq_k,
      int input_causal) {
    left_size = ori_left_size;
    right_size = input_causal == 1 ? 0 : ori_right_size;

    is_causal = (right_size == 0 && left_size < 0);
    int max_seq = std::max(max_seq_q, max_seq_k);
    if (left_size < 0 && right_size >= 0) {
      left_size = max_seq;
    }
    if (left_size >= 0 && right_size < 0) {
      right_size = max_seq;
    }
    attn_mask_mode =
        is_causal ? CNNL_ATTN_MASK_CAUSAL_TOP_LEFT : CNNL_ATTN_MASK_NONE;
  }
};

struct MaskParamsBwd {
  bool is_causal;
  int left_size = -1;
  int right_size = -1;
  cnnlAttentionMaskMode_t attn_mask_mode;
  MaskParamsBwd(
      const int64_t ori_left_size,
      const int64_t ori_right_size,
      const int64_t max_seq_q,
      const int64_t max_seq_k,
      int input_causal) {
    left_size = ori_left_size;
    right_size = input_causal == 1 ? 0 : ori_right_size;

    is_causal = (right_size == 0 && left_size < 0);
    int max_seq = std::max(max_seq_q, max_seq_k);
    if (left_size < 0 && right_size >= 0) {
      left_size = max_seq;
    }
    if (left_size >= 0 && right_size < 0) {
      right_size = max_seq;
    }
    if (left_size < 0 && right_size < 0) {
      left_size = max_seq;
      right_size = max_seq;
    }
    attn_mask_mode = is_causal ? CNNL_ATTN_MASK_CAUSAL_TOP_LEFT
                               : CNNL_ATTN_MASK_LOCAL_TOP_LEFT;
  }
};

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
    const bool return_softmax) {
  auto handle = getCurrentHandle();
  // check type
  auto q_dtype = q.dtype();
  TORCH_CHECK(
      q_dtype == at::kHalf || q_dtype == at::kBFloat16,
      "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(
      out.dtype() == q_dtype, "Output must have the same dtype as inputs");

  // check device
  TORCH_CHECK(
      q.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      k.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      v.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      out.device().is_privateuseone(), "Output tensor must be on MLU device");

  // check stride
  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      out.stride(-1) == 1, "Output tensor must have contiguous last dimension");

  // check shape
  const auto sizes = q.sizes();
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = sizes[0];
  const int num_heads = sizes[1];
  const int head_size = sizes[2];
  const int num_heads_k = k.size(1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention forward only supports head dimension at most 256");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");
  CHECK_SHAPE(out, total_q, num_heads, head_size);

  auto opts = q.options();
  cnnlFlashAttentionDescriptor_t fa_desc;
  TORCH_CNNL_CHECK(cnnlCreateFlashAttentionDescriptor(&fa_desc));
  int32_t dilation = 1;
  int window_size_left = -1;
  int window_size_right = -1;

  auto compute_dtype = CNNL_DTYPE_FLOAT;
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  MaskParamsFwd mask_params{
      window_size_left,
      window_size_right,
      max_seqlen_q,
      max_seqlen_k,
      (int)is_causal};

  TORCH_CNNL_CHECK(cnnlSetFlashAttentionSlidingWindowSize(
      fa_desc, mask_params.left_size, mask_params.right_size, dilation));
  TORCH_CNNL_CHECK(cnnlSetFlashAttentionDescriptor_v2(
      fa_desc,
      compute_dtype,
      prefer,
      mask_params.attn_mask_mode,
      CNNL_ATTN_TENSOR_LAYOUT_PACKED,
      /*is_pack_mode = */ true,
      zero_tensors,
      return_softmax,
      max_seqlen_q,
      max_seqlen_k,
      p_dropout,
      softmax_scale));

  auto softmax_lse = at::empty({num_heads, total_q}, opts.dtype(at::kFloat));

  auto [query_desc, query_ptr] =
      GetTensorDescAndMluDataPtr(q, CNNL_LAYOUT_ARRAY);
  auto [key_desc, key_ptr] = GetTensorDescAndMluDataPtr(k, CNNL_LAYOUT_ARRAY);
  auto [value_desc, value_ptr] =
      GetTensorDescAndMluDataPtr(v, CNNL_LAYOUT_ARRAY);

  auto [csq_desc, csq_ptr] =
      GetTensorDescAndMluDataPtr(cu_seqlens_q, CNNL_LAYOUT_ARRAY);
  auto [csk_desc, csk_ptr] =
      GetTensorDescAndMluDataPtr(cu_seqlens_k, CNNL_LAYOUT_ARRAY);

  auto [out_desc, out_ptr] = GetTensorDescAndMluDataPtr(out, CNNL_LAYOUT_ARRAY);
  auto [softmax_lse_desc, softmax_lse_ptr] =
      GetTensorDescAndMluDataPtr(softmax_lse, CNNL_LAYOUT_ARRAY);

  at::Tensor dropout_mask;
  void* dropout_mask_ptr = nullptr;
  tensorDescPtr_t dropout_mask_desc = nullptr;
  if (return_softmax) {
    dropout_mask = at::zeros(
        {batch_size, num_heads, max_seqlen_q, max_seqlen_k},
        opts.dtype(at::kFloat));
    std::tie(dropout_mask_desc, dropout_mask_ptr) =
        GetTensorDescAndMluDataPtr(dropout_mask, CNNL_LAYOUT_ARRAY);
  }

  size_t fa_counter_offset = 0;
  TORCH_CNNL_CHECK(cnnlGetFlashAttentionGeneratedRandomNumbers(
      handle,
      fa_desc,
      query_desc.get(),
      value_desc.get(),
      csq_desc.get(),
      &fa_counter_offset));

  at::Tensor seed_t, offset_t;
  auto rng_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  auto rng_tensor = at::empty({2}, rng_opts.dtype(at::kLong));
  auto rng_state = reinterpret_cast<uint64_t*>(rng_tensor.data_ptr());
  int thread_num = 0;
  cnnlRandRngType_t rng_type = CNNL_RAND_RNG_PHILOX;
  TORCH_CNNL_CHECK(
      cnnlGetRandSimulateThreadNum_v2(handle, rng_type, &thread_num));
  auto counter_offset =
      getCounterOffset(fa_counter_offset, (int64_t)thread_num);
  if (p_dropout > 0.0) {
    auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(
        c10::nullopt, getDefaultMLUGenerator());
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    auto philox_state = gen_impl->philox_mlu_state(counter_offset);
    rng_state[0] = static_cast<size_t>(philox_state.seed_.val);
    rng_state[1] = static_cast<size_t>(philox_state.offset_.val);
    seed_t = at::scalar_tensor(
        at::Scalar(static_cast<int64_t>(philox_state.seed_.val)),
        at::dtype(at::kLong));
    offset_t = at::scalar_tensor(
        at::Scalar(static_cast<int64_t>(philox_state.offset_.val)),
        at::dtype(at::kLong));
  } else {
    seed_t = at::empty({}, at::dtype(at::kLong));
    offset_t = at::empty({}, at::dtype(at::kLong));
  }
  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetFlashAttentionForwardWorkspaceSize_v2(
      handle,
      fa_desc,
      query_desc.get(),
      key_desc.get(),
      value_desc.get(),
      csq_desc.get(),
      csk_desc.get(),
      nullptr,
      nullptr,
      nullptr,
      softmax_lse_desc.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  TORCH_CNNL_CHECK(cnnlFlashAttentionForward_v2(
      handle,
      fa_desc,
      query_desc.get(),
      query_ptr,
      key_desc.get(),
      key_ptr,
      value_desc.get(),
      value_ptr,
      csq_desc.get(),
      csq_ptr,
      csk_desc.get(),
      csk_ptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      rng_state,
      workspace_ptr.get(),
      workspace_size,
      dropout_mask_desc.get(),
      dropout_mask_ptr,
      softmax_lse_desc.get(),
      softmax_lse_ptr,
      out_desc.get(),
      out_ptr));

  cnnlDestroyFlashAttentionDescriptor(fa_desc);
  int max_seqlen_q_lse = ((max_seqlen_q + 16 - 1) / 16) * 16;
  softmax_lse =
      softmax_lse.view({num_heads, batch_size, max_seqlen_q}).transpose(0, 1);
  auto softmax_lse_paded = at::pad_symint(
      softmax_lse, {c10::SymInt{0}, (max_seqlen_q_lse - max_seqlen_q)});
  return {softmax_lse_paded, seed_t, offset_t, dropout_mask};
}

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
    const at::Tensor& philox_offset) {
  auto handle = getCurrentHandle();

  auto q_dtype = q.dtype();
  TORCH_CHECK(
      q_dtype == at::kHalf || q_dtype == at::kBFloat16,
      "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(
      dout.dtype() == q_dtype, "query and dout must have the same dtype");
  TORCH_CHECK(dq.dtype() == q_dtype, "query and dq must have the same dtype");
  TORCH_CHECK(dk.dtype() == q_dtype, "query and dk must have the same dtype");
  TORCH_CHECK(dv.dtype() == q_dtype, "query and dv must have the same dtype");
  TORCH_CHECK(
      cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(
      cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must have dtype int32");

  TORCH_CHECK(
      q.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      k.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      v.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      out.device().is_privateuseone(), "out tensor must be on MLU device");
  TORCH_CHECK(
      dout.device().is_privateuseone(), "dout tensor must be on MLU device");
  TORCH_CHECK(
      softmax_lse.device().is_privateuseone(),
      "softmax_lse tensor must be on MLU device");
  TORCH_CHECK(
      cu_seqlens_q.device().is_privateuseone(),
      "cu_seqlens_q must be on MLU device");
  TORCH_CHECK(
      cu_seqlens_k.device().is_privateuseone(),
      "cu_seqlens_k must be on MLU device");

  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      out.stride(-1) == 1, "out tensor must have contiguous last dimension");
  TORCH_CHECK(
      dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
  TORCH_CHECK(
      dq.stride(-1) == 1, "dq tensor must have contiguous last dimension");
  TORCH_CHECK(
      dk.stride(-1) == 1, "dk tensor must have contiguous last dimension");
  TORCH_CHECK(
      dv.stride(-1) == 1, "dv tensor must have contiguous last dimension");
  TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be contiguous");
  TORCH_CHECK(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");

  const auto sizes = q.sizes();
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = sizes[0];
  const int num_heads = sizes[1];
  const int head_size = sizes[2];
  const int total_k = k.size(0);
  const int num_heads_k = k.size(1);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(
      head_size <= 256,
      "FlashAttention backward only supports head dimension at most 256");
  TORCH_CHECK(
      num_heads % num_heads_k == 0,
      "Number of heads in key/value must divide number of heads in query");
  CHECK_SHAPE(out, total_q, num_heads, head_size);
  CHECK_SHAPE(dq, total_q, num_heads, head_size);
  CHECK_SHAPE(dk, total_k, num_heads, head_size);
  CHECK_SHAPE(dv, total_k, num_heads, head_size);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  int window_size_right = -1;
  int window_size_left = -1;

  at::Tensor softmax_lse_tmp = cnnl_contiguous(
      softmax_lse.slice_symint(-1, 0, max_seqlen_q).transpose(0, 1));
  at::Tensor softmax_lse_view =
      softmax_lse_tmp.reshape({num_heads, batch_size * max_seqlen_q});

  size_t* rng_tensor = nullptr;

  auto rng_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  auto rng_state = at::empty({2}, rng_opts.dtype(at::kLong));
  rng_tensor = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

  bool is_dropout = p_dropout > 0.0;
  if (is_dropout) {
    rng_state[0] = philox_seed;
    rng_state[1] = philox_offset;
  }

  cnnlFlashAttentionDescriptor_t fa_desc;
  TORCH_CNNL_CHECK(cnnlCreateFlashAttentionDescriptor(&fa_desc));
  int32_t dilation = 1;

  auto compute_dtype = CNNL_DTYPE_FLOAT;
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MaskParamsBwd mask_params{
      window_size_left,
      window_size_right,
      max_seqlen_q,
      max_seqlen_k,
      (int)is_causal};
  TORCH_CNNL_CHECK(cnnlSetFlashAttentionSlidingWindowSize(
      fa_desc, mask_params.left_size, mask_params.right_size, dilation));
  TORCH_CNNL_CHECK(cnnlSetFlashAttentionBackwardDescriptor_v2(
      fa_desc,
      compute_dtype,
      prefer,
      mask_params.attn_mask_mode,
      CNNL_ATTN_TENSOR_LAYOUT_PACKED,
      /*is_pack_mode = */ true,
      /*is_out_zero = */ false,
      /*is_store_softmax_d = */ false,
      max_seqlen_q,
      max_seqlen_k,
      p_dropout,
      softmax_scale));

  auto [diff_out_desc, diff_out_ptr] =
      GetTensorDescAndMluDataPtr(dout, CNNL_LAYOUT_ARRAY);
  auto [query_desc, query_ptr] =
      GetTensorDescAndMluDataPtr(q, CNNL_LAYOUT_ARRAY);
  auto [key_desc, key_ptr] = GetTensorDescAndMluDataPtr(k, CNNL_LAYOUT_ARRAY);
  auto [value_desc, value_ptr] =
      GetTensorDescAndMluDataPtr(v, CNNL_LAYOUT_ARRAY);
  auto [fwd_out_desc, fwd_out_ptr] =
      GetTensorDescAndMluDataPtr(out, CNNL_LAYOUT_ARRAY);
  auto [csq_desc, csq_ptr] =
      GetTensorDescAndMluDataPtr(cu_seqlens_q, CNNL_LAYOUT_ARRAY);
  auto [csk_desc, csk_ptr] =
      GetTensorDescAndMluDataPtr(cu_seqlens_k, CNNL_LAYOUT_ARRAY);

  auto [softmax_lse_desc, softmax_lse_ptr] =
      GetTensorDescAndMluDataPtr(softmax_lse_view, CNNL_LAYOUT_ARRAY);
  auto [diff_query_desc, diff_query_ptr] =
      GetTensorDescAndMluDataPtr(dq, CNNL_LAYOUT_ARRAY);
  auto [diff_key_desc, diff_key_ptr] =
      GetTensorDescAndMluDataPtr(dk, CNNL_LAYOUT_ARRAY);
  auto [diff_value_desc, diff_value_ptr] =
      GetTensorDescAndMluDataPtr(dv, CNNL_LAYOUT_ARRAY);

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetFlashAttentionBackwardWorkspaceSize_v2(
      handle,
      fa_desc,
      query_desc.get(),
      key_desc.get(),
      value_desc.get(),
      csq_desc.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlFlashAttentionBackward_v2(
      handle,
      fa_desc,
      diff_out_desc.get(),
      diff_out_ptr,
      query_desc.get(),
      query_ptr,
      key_desc.get(),
      key_ptr,
      value_desc.get(),
      value_ptr,
      fwd_out_desc.get(),
      fwd_out_ptr,
      softmax_lse_desc.get(),
      softmax_lse_ptr,
      csq_desc.get(),
      csq_ptr,
      csk_desc.get(),
      csk_ptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      rng_tensor,
      workspace_ptr.get(),
      workspace_size,
      diff_query_desc.get(),
      diff_query_ptr,
      diff_key_desc.get(),
      diff_key_ptr,
      diff_value_desc.get(),
      diff_value_ptr,
      /*dropout_mask_desc = */ nullptr,
      /*dropout_mask = */ nullptr,
      /*softmax_d_desc = */ nullptr,
      /*softmax_d = */ nullptr));
  cnnlDestroyFlashAttentionDescriptor(fa_desc);
  return {dq, dk, dv, at::Tensor()};
}

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
    const float scale) {
  auto handle = getCurrentHandle();
  // check type
  auto q_dtype = query.dtype();
  TORCH_CHECK(
      q_dtype == at::kHalf || q_dtype == at::kBFloat16,
      "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(key.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(
      value.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(
      output.dtype() == q_dtype, "Output must have the same dtype as inputs");
  TORCH_CHECK(
      cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(
      cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must have dtype int32");
  // check device
  TORCH_CHECK(
      query.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      key.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      value.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      output.device().is_privateuseone(),
      "Output tensor must be on MLU device");
  TORCH_CHECK(
      cu_seqlens_q.device().is_privateuseone(),
      "cu_seqlens_q must be on MLU device");
  TORCH_CHECK(
      cu_seqlens_k.device().is_privateuseone(),
      "cu_seqlens_k must be on MLU device");
  // check stride
  TORCH_CHECK(
      query.stride(-1) == 1,
      "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      key.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      value.stride(-1) == 1,
      "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      output.stride(-1) == 1,
      "Output tensor must have contiguous last dimension");
  TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be contiguous");
  TORCH_CHECK(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");
  // check shape
  const auto sizes = query.sizes();
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = sizes[0];
  const int num_heads = sizes[1];
  const int embedding_value = value.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  CHECK_SHAPE(output, total_q, num_heads, embedding_value);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  cnnlFlashAttentionDescriptor_t me_desc;
  TORCH_CNNL_CHECK(cnnlCreateFlashAttentionDescriptor(&me_desc));
  int32_t dilation = 1;
  int window_size_left = -1;
  int window_size_right = -1;

  auto compute_dtype = CNNL_DTYPE_FLOAT;
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;

  MaskParamsFwd mask_params{
      window_size_left,
      window_size_right,
      max_seqlen_q,
      max_seqlen_k,
      (int)custom_mask_type};
  TORCH_CNNL_CHECK(cnnlSetFlashAttentionSlidingWindowSize(
      me_desc, mask_params.left_size, mask_params.right_size, dilation));

  TORCH_CNNL_CHECK(cnnlSetFlashAttentionDescriptor_v2(
      me_desc,
      compute_dtype,
      prefer,
      mask_params.attn_mask_mode,
      CNNL_ATTN_TENSOR_LAYOUT_PACKED,
      /*is_pack_mode = */ true,
      false,
      false,
      max_seqlen_q,
      max_seqlen_k,
      dropout_p,
      scale));

  auto [query_desc, query_ptr] =
      GetTensorDescAndMluDataPtr(query, CNNL_LAYOUT_ARRAY);
  auto [key_desc, key_ptr] = GetTensorDescAndMluDataPtr(key, CNNL_LAYOUT_ARRAY);
  auto [value_desc, value_ptr] =
      GetTensorDescAndMluDataPtr(value, CNNL_LAYOUT_ARRAY);

  auto [csq_desc, csq_ptr] =
      GetTensorDescAndMluDataPtr(cu_seqlens_q, CNNL_LAYOUT_ARRAY);
  auto [csk_desc, csk_ptr] =
      GetTensorDescAndMluDataPtr(cu_seqlens_k, CNNL_LAYOUT_ARRAY);
  auto [out_desc, out_ptr] =
      GetTensorDescAndMluDataPtr(output, CNNL_LAYOUT_ARRAY);

  void* bias_ptr = nullptr;
  tensorDescPtr_t bias_desc = nullptr;
  if (bias.has_value()) {
    std::tie(bias_desc, bias_ptr) =
        GetTensorDescAndMluDataPtr(*bias, CNNL_LAYOUT_ARRAY);
  }

  at::Tensor logsumexp;
  void* logsumexp_ptr = nullptr;
  tensorDescPtr_t logsumexp_desc = nullptr;
  if (compute_log_sumexp) {
    auto opts = query.options();
    logsumexp = at::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    std::tie(logsumexp_desc, logsumexp_ptr) =
        GetTensorDescAndMluDataPtr(logsumexp, CNNL_LAYOUT_ARRAY);
  }

  size_t me_counter_offset = 0;
  TORCH_CNNL_CHECK(cnnlGetFlashAttentionGeneratedRandomNumbers(
      handle,
      me_desc,
      query_desc.get(),
      value_desc.get(),
      csq_desc.get(),
      &me_counter_offset));
  at::Tensor seed_t, offset_t;
  auto rng_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  auto rng_tensor = at::empty({2}, rng_opts.dtype(at::kLong));
  auto rng_state = reinterpret_cast<uint64_t*>(rng_tensor.data_ptr());
  int thread_num = 0;
  cnnlRandRngType_t rng_type = CNNL_RAND_RNG_PHILOX;
  TORCH_CNNL_CHECK(
      cnnlGetRandSimulateThreadNum_v2(handle, rng_type, &thread_num));
  auto counter_offset =
      getCounterOffset(me_counter_offset, (int64_t)thread_num);
  if (dropout_p > 0.0) {
    auto gen_impl = at::get_generator_or_default<MLUGeneratorImpl>(
        c10::nullopt, getDefaultMLUGenerator());
    std::lock_guard<std::mutex> lock(gen_impl->mutex_);
    auto philox_state = gen_impl->philox_mlu_state(counter_offset);
    rng_state[0] = static_cast<size_t>(philox_state.seed_.val);
    rng_state[1] = static_cast<size_t>(philox_state.offset_.val);
    seed_t = at::scalar_tensor(
        at::Scalar(static_cast<int64_t>(philox_state.seed_.val)),
        at::dtype(at::kLong));
    offset_t = at::scalar_tensor(
        at::Scalar(static_cast<int64_t>(philox_state.offset_.val)),
        at::dtype(at::kLong));
  } else {
    seed_t = at::empty({}, at::dtype(at::kLong));
    offset_t = at::empty({}, at::dtype(at::kLong));
  }

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetFlashAttentionForwardWorkspaceSize_v2(
      handle,
      me_desc,
      query_desc.get(),
      key_desc.get(),
      value_desc.get(),
      csq_desc.get(),
      csk_desc.get(),
      nullptr,
      bias_desc.get(),
      nullptr,
      logsumexp_desc.get(),
      &workspace_size));

  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);
  TORCH_CNNL_CHECK(cnnlFlashAttentionForward_v2(
      handle,
      me_desc,
      query_desc.get(),
      query_ptr,
      key_desc.get(),
      key_ptr,
      value_desc.get(),
      value_ptr,
      csq_desc.get(),
      csq_ptr,
      csk_desc.get(),
      csk_ptr,
      nullptr,
      nullptr,
      bias_desc.get(),
      bias_ptr,
      nullptr,
      nullptr,
      rng_state,
      workspace_ptr.get(),
      workspace_size,
      nullptr,
      nullptr,
      logsumexp_desc.get(),
      logsumexp_ptr,
      out_desc.get(),
      out_ptr));

  cnnlDestroyFlashAttentionDescriptor(me_desc);
  return {logsumexp, seed_t, offset_t};
}

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
    const int64_t custom_mask_type) {
  auto handle = getCurrentHandle();
  // check dtype
  auto q_dtype = q.dtype();
  TORCH_CHECK(
      q_dtype == at::kHalf || q_dtype == at::kBFloat16,
      "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
  TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
  TORCH_CHECK(
      dout.dtype() == q_dtype, "query and dout must have the same dtype");
  TORCH_CHECK(dq.dtype() == q_dtype, "query and dq must have the same dtype");
  TORCH_CHECK(dk.dtype() == q_dtype, "query and dk must have the same dtype");
  TORCH_CHECK(dv.dtype() == q_dtype, "query and dv must have the same dtype");
  TORCH_CHECK(
      cu_seqlens_q.dtype() == at::kInt, "cu_seqlens_q must have dtype int32");
  TORCH_CHECK(
      cu_seqlens_k.dtype() == at::kInt, "cu_seqlens_k must have dtype int32");
  // check device
  TORCH_CHECK(
      q.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      k.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      v.device().is_privateuseone(), "Input tensor must be on MLU device");
  TORCH_CHECK(
      out.device().is_privateuseone(), "out tensor must be on MLU device");
  TORCH_CHECK(
      dout.device().is_privateuseone(), "dout tensor must be on MLU device");
  TORCH_CHECK(
      softmax_lse.device().is_privateuseone(),
      "softmax_lse tensor must be on MLU device");
  TORCH_CHECK(
      cu_seqlens_q.device().is_privateuseone(),
      "cu_seqlens_q must be on MLU device");
  TORCH_CHECK(
      cu_seqlens_k.device().is_privateuseone(),
      "cu_seqlens_k must be on MLU device");
  // check stride
  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      out.stride(-1) == 1, "out tensor must have contiguous last dimension");
  TORCH_CHECK(
      dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
  TORCH_CHECK(
      dq.stride(-1) == 1, "dq tensor must have contiguous last dimension");
  TORCH_CHECK(
      dk.stride(-1) == 1, "dk tensor must have contiguous last dimension");
  TORCH_CHECK(
      dv.stride(-1) == 1, "dv tensor must have contiguous last dimension");
  TORCH_CHECK(cu_seqlens_q.is_contiguous(), "cu_seqlens_q must be contiguous");
  TORCH_CHECK(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");
  // check shape
  const auto sizes = q.sizes();
  const int batch_size = cu_seqlens_q.numel() - 1;
  const int total_q = sizes[0];
  const int num_heads = sizes[1];
  const int embedding = sizes[2];
  const int embedding_value = v.size(2);
  const int total_k = k.size(0);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  CHECK_SHAPE(out, total_q, num_heads, embedding_value);
  CHECK_SHAPE(dq, total_q, num_heads, embedding);
  CHECK_SHAPE(dk, total_k, num_heads, embedding);
  CHECK_SHAPE(dv, total_k, num_heads, embedding_value);
  CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
  CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

  int window_size_right = -1;
  int window_size_left = -1;

  size_t* rng_tensor = nullptr;

  auto rng_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
  auto rng_state = at::empty({2}, rng_opts.dtype(at::kLong));
  rng_tensor = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

  bool is_dropout = p_dropout > 0.0;
  if (is_dropout) {
    rng_state[0] = philox_seed;
    rng_state[1] = philox_offset;
  }

  cnnlFlashAttentionDescriptor_t me_desc;
  TORCH_CNNL_CHECK(cnnlCreateFlashAttentionDescriptor(&me_desc));
  int32_t dilation = 1;

  auto compute_dtype = CNNL_DTYPE_FLOAT;
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MaskParamsBwd mask_params{
      window_size_left,
      window_size_right,
      max_seqlen_q,
      max_seqlen_k,
      (int)custom_mask_type};
  TORCH_CNNL_CHECK(cnnlSetFlashAttentionSlidingWindowSize(
      me_desc, mask_params.left_size, mask_params.right_size, dilation));

  TORCH_CNNL_CHECK(cnnlSetFlashAttentionBackwardDescriptor_v2(
      me_desc,
      compute_dtype,
      prefer,
      mask_params.attn_mask_mode,
      CNNL_ATTN_TENSOR_LAYOUT_PACKED,
      /*is_pack_mode = */ true,
      /*is_out_zero = */ false,
      /*is_store_softmax_d = */ false,
      max_seqlen_q,
      max_seqlen_k,
      p_dropout,
      softmax_scale));

  auto [diff_out_desc, diff_out_ptr] =
      GetTensorDescAndMluDataPtr(dout, CNNL_LAYOUT_ARRAY);
  auto [query_desc, query_ptr] =
      GetTensorDescAndMluDataPtr(q, CNNL_LAYOUT_ARRAY);
  auto [key_desc, key_ptr] = GetTensorDescAndMluDataPtr(k, CNNL_LAYOUT_ARRAY);
  auto [value_desc, value_ptr] =
      GetTensorDescAndMluDataPtr(v, CNNL_LAYOUT_ARRAY);
  auto [fwd_out_desc, fwd_out_ptr] =
      GetTensorDescAndMluDataPtr(out, CNNL_LAYOUT_ARRAY);
  auto [csq_desc, csq_ptr] =
      GetTensorDescAndMluDataPtr(cu_seqlens_q, CNNL_LAYOUT_ARRAY);
  auto [csk_desc, csk_ptr] =
      GetTensorDescAndMluDataPtr(cu_seqlens_k, CNNL_LAYOUT_ARRAY);

  auto [softmax_lse_desc, softmax_lse_ptr] =
      GetTensorDescAndMluDataPtr(softmax_lse, CNNL_LAYOUT_ARRAY);
  auto [diff_query_desc, diff_query_ptr] =
      GetTensorDescAndMluDataPtr(dq, CNNL_LAYOUT_ARRAY);
  auto [diff_key_desc, diff_key_ptr] =
      GetTensorDescAndMluDataPtr(dk, CNNL_LAYOUT_ARRAY);
  auto [diff_value_desc, diff_value_ptr] =
      GetTensorDescAndMluDataPtr(dv, CNNL_LAYOUT_ARRAY);

  void* bias_ptr = nullptr;
  tensorDescPtr_t bias_desc = nullptr;
  if (bias.has_value()) {
    std::tie(bias_desc, bias_ptr) =
        GetTensorDescAndMluDataPtr(*bias, CNNL_LAYOUT_ARRAY);
  }

  size_t workspace_size = 0;
  TORCH_CNNL_CHECK(cnnlGetFlashAttentionBackwardWorkspaceSize_v2(
      handle,
      me_desc,
      query_desc.get(),
      key_desc.get(),
      value_desc.get(),
      csq_desc.get(),
      &workspace_size));
  auto workspace_ptr =
      torch_mlu::MLUCachingAllocator::get()->allocate(workspace_size);

  TORCH_CNNL_CHECK(cnnlFlashAttentionBackward_v2(
      handle,
      me_desc,
      diff_out_desc.get(),
      diff_out_ptr,
      query_desc.get(),
      query_ptr,
      key_desc.get(),
      key_ptr,
      value_desc.get(),
      value_ptr,
      fwd_out_desc.get(),
      fwd_out_ptr,
      softmax_lse_desc.get(),
      softmax_lse_ptr,
      csq_desc.get(),
      csq_ptr,
      csk_desc.get(),
      csk_ptr,
      nullptr,
      nullptr,
      bias_desc.get(),
      bias_ptr,
      rng_tensor,
      workspace_ptr.get(),
      workspace_size,
      diff_query_desc.get(),
      diff_query_ptr,
      diff_key_desc.get(),
      diff_key_ptr,
      diff_value_desc.get(),
      diff_value_ptr,
      /*dropout_mask_desc = */ nullptr,
      /*dropout_mask = */ nullptr,
      /*softmax_d_desc = */ nullptr,
      /*softmax_d = */ nullptr));
  cnnlDestroyFlashAttentionDescriptor(me_desc);
  return {dq, dk, dv};
}

} // namespace ops
} // namespace torch_mlu