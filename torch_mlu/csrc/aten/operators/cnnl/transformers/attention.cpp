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

#include <tuple>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Logging.h>
#include <ATen/TensorOptions.h>
#include <ATen/ops/_transform_bias_rescale_qkv.h>
#include "sdp_utils.h"
#include "aten/operators/cnnl/cnnl_kernel.h"
#include "aten/operators/cnnl/internal/cnnl_internal.h"

namespace torch_mlu {
namespace ops {

#define CHECK_SHAPE(x, ...)                        \
  TORCH_CHECK(                                     \
      x.sizes() == at::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")

void debug_assert_shape(int line, const Tensor& t, c10::IntArrayRef shape) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (size_t)t.dim() == shape.size(),
      "(called from line ",
      line,
      ") ",
      "expected ",
      shape.size(),
      "-D tensor but got ",
      t.dim());
  if (t.is_nested()) {
    return;
  }
  for (auto idx : c10::irange(shape.size())) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        shape[idx] == 0 || t.sizes()[idx] == shape[idx],
        "(called from line ",
        line,
        ") ",
        "expected dim ",
        idx,
        " to be ",
        shape[idx],
        " but got ",
        t.sizes()[idx]);
  }
}

Tensor gemm_nt(const Tensor& self, const Tensor& other) {
  return at::native::matmul(self, other.t());
}

at::Tensor bmm_nt(const at::Tensor& a, const at::Tensor& b) {
  auto a_ = a.view({a.size(0) * a.size(1), a.size(2), a.size(3)});
  auto b_ = b.view({b.size(0) * b.size(1), b.size(2), b.size(3)});
  auto bt_ = b_.transpose(2, 1);
  auto c_ = at::bmm(a_, bt_);
  return c_.view({a.size(0), a.size(1), a.size(2), b.size(2)});
}

at::Tensor bmm_nn(at::Tensor& out, const at::Tensor& a, const at::Tensor& b) {
  const std::array<int64_t, 3> newAShape = {
      a.sizes()[0] * a.sizes()[1], a.sizes()[2], a.sizes()[3]};
  auto a_ = a.view(newAShape);
  const std::array<int64_t, 3> newBShape = {
      b.sizes()[0] * b.sizes()[1], b.sizes()[2], b.sizes()[3]};
  auto b_ = b.view(newBShape);
  auto out_ = out.reshape({newAShape[0], newAShape[1], newBShape[2]});
  auto c_ = at::bmm_out(out_, a_, b_);
  return c_.view({a.size(0), a.size(1), a.size(2), b.size(3)});
}

Tensor transform_0213(const at::Tensor& a) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.size(1));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.size(3));
  return a.permute({0, 2, 1, 3})
      .contiguous()
      .view({a.size(0), a.size(2), a.size(1) * a.size(3)});
}

at::Tensor transform0213_gemm_nt_bias(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& query) {
  const Tensor a_0213 = transform_0213(a);
  auto a_ = a_0213.view({a_0213.size(0) * a_0213.size(1), a_0213.size(2)});
  auto r_ = at::native::linear(a_, b, c);
  return r_.view({a_0213.size(0), a_0213.size(1), r_.size(1)});
}

at::Tensor qkv_projection(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const int64_t embed_dim,
    const at::Tensor& qkv_weight) {
  // shape: [B, T, 3 x D]
  at::Tensor qkv;

  if (key.is_same(value)) {
    if (query.is_same(key)) {
      // self-attention
      qkv = gemm_nt(query, qkv_weight);
    } else {
      // encoder-decoder attention
      // TODO: is there a more efficient way to set this up?
      // TODO: can we stay nested insted of using cat? Probably just make a
      // NestedTensor out of the matmul results or something?
      auto q_kv_weight_s = at::native::split_with_sizes(
          qkv_weight, {embed_dim, embed_dim * 2}, 0);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          q_kv_weight_s.size() == 2,
          "expected split to produce 2 tensors but it produced ",
          q_kv_weight_s.size());
      auto q = gemm_nt(query, q_kv_weight_s[0]);
      auto kv = gemm_nt(key, q_kv_weight_s[1]);
      qkv = at::cat({std::move(q), std::move(kv)}, 2);
    }
  } else {
    auto q_k_v_weight_s = at::native::chunk(qkv_weight, 3, 0);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        q_k_v_weight_s.size() == 3,
        "expected chunk to produce 3 tensors but it produced ",
        q_k_v_weight_s.size());
    // TODO: can we stay nested instead of using cat?
    auto q = gemm_nt(query, q_k_v_weight_s[0]);
    auto k = gemm_nt(key, q_k_v_weight_s[1]);
    auto v = gemm_nt(value, q_k_v_weight_s[2]);
    qkv = at::cat({std::move(q), std::move(k), std::move(v)}, 2);
  }

  return qkv;
}

Tensor masked_softmax(
    Tensor& attn_scores,
    c10::optional<Tensor> attn_mask,
    const Tensor& query,
    c10::optional<int64_t> mask_type) {
  if (query.is_nested() && !attn_mask) {
    return at::_nested_tensor_softmax_with_shape(attn_scores, query);
  }
  if (attn_mask && attn_mask->dtype() != at::kBool) {
    attn_mask = attn_mask->to(at::kBool);
  }
  if (attn_mask) {
    return _masked_softmax(
        attn_scores, *attn_mask, attn_scores.dim() - 1, mask_type);
  } else {
    return _softmax_out(attn_scores, attn_scores, attn_scores.dim() - 1, false);
  }
}

std::tuple<at::Tensor, at::Tensor> cnnl__native_multi_head_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    int64_t embed_dim,
    int64_t num_head,
    const at::Tensor& qkv_weight,
    const at::Tensor& qkv_bias,
    const at::Tensor& proj_weight,
    const at::Tensor& proj_bias,
    const c10::optional<at::Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    c10::optional<int64_t> mask_type) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]
  TORCH_CHECK(
      !mask || !query.is_nested(),
      "NestedTensor with mask is not supported yet");
  const auto D = embed_dim;
  TORCH_CHECK(
      query.dim() == 3, "expected 3-D `query`, got ", query.dim(), "-D tensor");
  TORCH_CHECK(
      query.is_nested() || query.sizes()[2] == embed_dim,
      "passed-in embed_dim ",
      embed_dim,
      " didn't match last dim of query ",
      query.sizes()[2]);
  TORCH_CHECK(
      key.dim() == 3, "expected 3-D `key`, got ", key.dim(), "-D tensor");
  TORCH_CHECK(
      value.dim() == 3, "expected 3-D `value`, got ", value.dim(), "-D tensor");
  TORCH_CHECK(
      query.is_nested() || key.is_nested() || value.is_nested() ||
          (query.sizes() == key.sizes() && key.sizes() == value.sizes()),
      "expected `query`/`key`/`value` shapes to match");
  TORCH_CHECK(
      qkv_weight.dim() == 2,
      "expected 2-D `qkv_weight`, got ",
      qkv_weight.dim(),
      "-D tensor");
  TORCH_CHECK(
      D * 3 == qkv_weight.sizes()[0],
      "expected `qkv_weight` first dim to be 3x embed_dim");
  TORCH_CHECK(
      D == qkv_weight.sizes()[1],
      "expected `qkv_weight` second dim to be embed_Dim");
  TORCH_CHECK(
      qkv_bias.dim() == 1,
      "expected 1-D `qkv_bias`, got ",
      qkv_bias.dim(),
      "-D tensor");
  TORCH_CHECK(
      qkv_bias.sizes()[0] == 3 * D,
      "expected `qkv_bias` first dim and first dim of query to be equal");
  TORCH_CHECK(
      D % num_head == 0, "`embed_dim` must divide evenly by `num_heads`");

#ifndef NDEBUG
  const auto B = query.is_nested()
      ? at::native::get_nested_tensor_impl(query)->get_nested_sizes().size(0)
      : query.sizes()[0];
  auto T = query.is_nested() ? 0 : query.sizes()[1];
#endif
  const auto dim_per_head = D / num_head;
  if ((query.is_same(key) && key.is_same(value)) && dim_per_head % 8 == 0 &&
      !need_weights) {
    auto q =
        query.view({query.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto k =
        key.view({key.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto v =
        value.view({value.size(0), -1, num_head, dim_per_head}).transpose(1, 2);

    sdp::sdp_params kernel_params{q, k, v, mask, 0.0, false};
    auto backend = torch_mlu::sdp::select_sdp_backend(kernel_params);

    bool no_seq_len_1_nested = query.is_nested()
        ? sdp::check_for_seq_len_1_nested_tensor(kernel_params, false)
        : true;

    if (!mask.has_value() && no_seq_len_1_nested &&
        (backend == sdp::SDPBackend::flash_attention ||
         backend == sdp::SDPBackend::efficient_attention)) {
      auto x = at::linear(query, qkv_weight, qkv_bias);
      auto chunks = x.chunk(3, -1);
      auto x_size_0 = x.size(0);

      chunks[0] = (chunks[0].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[1] = (chunks[1].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[2] = (chunks[2].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      auto y = at::scaled_dot_product_attention(
          chunks[0], chunks[1], chunks[2], mask, 0.0, false, c10::nullopt);

      auto past_sdp = y.transpose(1, 2).reshape({x_size_0, -1, embed_dim});
      return std::make_tuple(
          at::linear(past_sdp, proj_weight, proj_bias), Tensor());
    }
    // Returned math or error lets not use it
  }

  auto qkv = qkv_projection(query, key, value, embed_dim, qkv_weight);
  if (!qkv.is_nested() && qkv.numel() == 0) {
    if (query.is_nested()) {
      return std::make_tuple(Tensor(), Tensor());
    }
    return std::make_tuple(at::empty_like(query), Tensor());
  }
#ifndef NDEBUG
  if (!query.is_nested() || !qkv.is_nested()) {
    if (query.is_nested()) {
      T = qkv.size(1);
    }
    debug_assert_shape(__LINE__, qkv, {B, T, 3 * D});
  }
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  if (!qkv.is_nested()) {
    std::cerr << "qkv: " << qkv << std::endl;
  }
#endif

  // shape: 3 x [B, num_head, T, dim_per_head]
  auto q_k_v = _transform_bias_rescale_qkv(qkv, qkv_bias, num_head);
  qkv = Tensor(); // Not used any more, allow free
  auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);
#ifndef NDEBUG
  debug_assert_shape(__LINE__, q, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, k, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, v, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "q: " << q << std::endl;
  std::cerr << "k: " << k << std::endl;
  std::cerr << "v: " << v << std::endl;
#endif

  // shape: [B, num_head, T, T]
  auto qkt = bmm_nt(q, k);
#ifndef NDEBUG
  debug_assert_shape(__LINE__, qkt, {B, num_head, T, T});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "qkt: " << qkt << std::endl;
#endif

  qkt = masked_softmax(qkt, mask, query, mask_type);
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "qkt after softmax: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, dim_per_head]
  // reuse storage for q; we're done with it
  auto attn_ctx = bmm_nn(q, qkt, v);
  // qkv is not dead; we just reused storage for q!
  if (!need_weights) {
    qkt = Tensor();
  }
#ifndef NDEBUG
  debug_assert_shape(__LINE__, attn_ctx, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "attn_ctx: " << attn_ctx << std::endl;
#endif

  // shape: [B, T, D]
  // Fuse transform_0213 inside
  auto proj =
      transform0213_gemm_nt_bias(attn_ctx, proj_weight, proj_bias, query);
#ifndef NDEBUG
  debug_assert_shape(__LINE__, proj, {B, T, D});
#endif

  if (need_weights && average_attn_weights) {
    // weights are not needed for full transformer, so don't worry too
    // much about performance -- we implement this just to make use
    // cases that don't disable need_weights still get some speedup.
    qkt = qkt.sum(1);
    qkt /= num_head;
  }

  return std::make_tuple(std::move(proj), std::move(qkt));
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    int64_t,
    int64_t,
    at::Tensor,
    at::Tensor,
    at::Tensor>
cnnl__scaled_dot_product_flash_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("torch.sdpa.flash_attention");

  // ndim
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(2) == value.size(2));

  // Num heads
  TORCH_CHECK(query.size(1) == key.size(1));
  TORCH_CHECK(query.size(1) == value.size(1));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t head_dim = query.size(3);

  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);
  TORCH_CHECK(
      max_seqlen_batch_k == max_seqlen_batch_v,
      "Key and Value must have the same sequence length");

  // Query -> Query(Batch x Q_seq_len x Num_heads x Dim_per_head)
  // Key   -> Key(Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x  Num_heads x Dim_per_head)
  at::Tensor q_t = query.transpose(1, 2);
  at::Tensor k_t = key.transpose(1, 2);
  at::Tensor v_t = value.transpose(1, 2);

  at::Tensor cumulative_sequence_length_q = at::arange(
      0,
      (batch_size + 1) * max_seqlen_batch_q,
      max_seqlen_batch_q,
      c10::TensorOptions().device(at::kPrivateUse1).dtype(at::kInt));

  at::Tensor cumulative_sequence_length_k = at::arange(
      0,
      (batch_size + 1) * max_seqlen_batch_k,
      max_seqlen_batch_k,
      c10::TensorOptions().device(at::kPrivateUse1).dtype(at::kInt));

  int64_t Nnz_q{batch_size * max_seqlen_batch_q};
  int64_t Nnz_kv{batch_size * max_seqlen_batch_k};

  at::Tensor q_t_c = cnnl_contiguous(q_t);
  at::Tensor query_reshaped = q_t_c.view({Nnz_q, num_heads, head_dim});
  at::Tensor k_t_c = cnnl_contiguous(k_t);
  at::Tensor key_reshaped = k_t_c.view({Nnz_kv, num_heads, head_dim});
  at::Tensor v_t_c = cnnl_contiguous(v_t);
  at::Tensor value_reshaped = v_t_c.view({Nnz_kv, num_heads, head_dim});

  at::Tensor attention, log_sumexp, debug_attn_mask, philox_seed, philox_offset;
  std::tie(attention, log_sumexp, philox_seed, philox_offset, debug_attn_mask) =
      at::_flash_attention_forward(
          query_reshaped,
          key_reshaped,
          value_reshaped,
          cumulative_sequence_length_q,
          cumulative_sequence_length_k,
          max_seqlen_batch_q,
          max_seqlen_batch_k,
          dropout_p,
          is_causal,
          return_debug_mask,
          scale);
  // Reshape output to convert nnz to batch_size and seq_len
  attention =
      attention.view({batch_size, max_seqlen_batch_q, num_heads, head_dim})
          .transpose(1, 2);
  return std::make_tuple(
      attention,
      log_sumexp,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      philox_seed,
      philox_offset,
      debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl__flash_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& cumulative_sequence_length_q,
    const at::Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    c10::optional<double> scale) {
  const auto softmax_scale =
      sdp::calculate_scale(query, scale).as_float_unchecked();
  at::Tensor output = at::empty_like(query);

  bool is_dropout = dropout_p > 0.0;
  bool return_debug_mask_ = return_debug_mask && is_dropout;

  auto [logsumexp, philox_seed, philox_offset, debug_attn_mask] =
      cnnl_fa_fwd_internal(
          query,
          key,
          value,
          output,
          cumulative_sequence_length_q,
          cumulative_sequence_length_k,
          max_seqlen_batch_q,
          max_seqlen_batch_k,
          dropout_p,
          softmax_scale,
          false, /*zero_tensors = false for all calls here*/
          is_causal,
          return_debug_mask_ /*return_softmax (this is used for testing)*/
      );

  debug_attn_mask =
      return_debug_mask_ ? debug_attn_mask : at::empty({0}, query.options());

  return std::make_tuple(
      output, logsumexp, philox_seed, philox_offset, debug_attn_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
cnnl__scaled_dot_product_flash_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    c10::optional<double> scale) {
  if (!grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }
  // ndim
  TORCH_CHECK(query.dim() == grad_out.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 4);

  // batch size
  TORCH_CHECK(query.size(0) == grad_out.size(0));
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // seqlen
  TORCH_CHECK(key.size(2) == value.size(2));
  TORCH_CHECK(query.size(2) == grad_out.size(2));

  // Num heads
  TORCH_CHECK(query.size(1) == key.size(1));
  TORCH_CHECK(query.size(1) == value.size(1));
  TORCH_CHECK(query.size(1) == grad_out.size(1));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));
  TORCH_CHECK(value.size(3) == grad_out.size(3));

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t head_dim = query.size(3);

  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  int64_t Nnz_q{batch_size * max_q};
  int64_t Nnz_kv{batch_size * max_k};

  at::Tensor q_t_c = cnnl_contiguous(q_t);
  at::Tensor query_reshaped = q_t_c.view({Nnz_q, num_heads, head_dim});
  at::Tensor k_t_c = cnnl_contiguous(k_t);
  at::Tensor key_reshaped = k_t_c.view({Nnz_kv, num_heads, head_dim});
  at::Tensor v_t_c = cnnl_contiguous(v_t);
  at::Tensor value_reshaped = v_t_c.view({Nnz_kv, num_heads, head_dim});

  auto grad_out_reshaped =
      grad_out.transpose(1, 2).reshape({{Nnz_q, num_heads, head_dim}});
  auto out_reshaped = out.transpose(1, 2).reshape({Nnz_q, num_heads, head_dim});

  Tensor grad_q, grad_k, grad_v;
  std::tie(grad_q, grad_k, grad_v) = at::_flash_attention_backward(
      grad_out_reshaped,
      query_reshaped,
      key_reshaped,
      value_reshaped,
      out_reshaped,
      logsumexp,
      cum_seq_q,
      cum_seq_k,
      max_q,
      max_k,
      dropout_p,
      is_causal,
      philox_seed,
      philox_offset,
      scale);

  grad_q =
      grad_q.view({batch_size, max_q, num_heads, head_dim}).transpose(1, 2);
  grad_k =
      grad_k.view({batch_size, max_k, num_heads, head_dim}).transpose(1, 2);
  grad_v =
      grad_v.view({batch_size, max_k, num_heads, head_dim}).transpose(1, 2);

  return std::make_tuple(grad_q, grad_k, grad_v);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cnnl__flash_attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& cum_seq_q,
    const at::Tensor& cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    c10::optional<double> scale) {
  const auto softmax_scale =
      sdp::calculate_scale(query, scale).as_float_unchecked();

  auto contiguous_grad_out = grad_out.contiguous();
  auto contiguous_out = out.contiguous();
  Tensor dq = at::empty_like(query);
  Tensor dk = at::empty_like(key);
  Tensor dv = at::empty_like(value);

  // The kernel computes irregadless we will drop for this functions return
  Tensor grad_softmax;

  std::tie(dq, dk, dv, grad_softmax) = cnnl_fa_bwd_internal(
      contiguous_grad_out,
      query,
      key,
      value,
      contiguous_out,
      logsumexp,
      dq,
      dk,
      dv,
      cum_seq_q,
      cum_seq_k,
      max_q,
      max_k,
      dropout_p,
      softmax_scale,
      false, /*zero_tensors = false for all calls here*/
      is_causal,
      philox_seed,
      philox_offset);
  return std::make_tuple(dq, dk, dv);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl__scaled_dot_product_efficient_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  C10_LOG_API_USAGE_ONCE("torch.sdpa.mem_efficient_attention");

  // Query -> Query(Batch x Q_seq_len x Num_heads x Dim_per_head)
  // Key   -> Key(Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x  Num_heads x Dim_per_head)
  at::Tensor q_t = query.transpose(1, 2);
  at::Tensor k_t = key.transpose(1, 2);
  at::Tensor v_t = value.transpose(1, 2);

  sdp::CustomMaskType custom_mask_type = is_causal
      ? sdp::CustomMaskType::CausalFromTopLeft
      : sdp::CustomMaskType::NoCustomMask;

  auto [attention, log_sumexp, seed, offset] = at::_efficient_attention_forward(
      q_t,
      k_t,
      v_t,
      attn_bias,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      dropout_p /*dropout_p*/,
      static_cast<int64_t>(custom_mask_type),
      compute_log_sumexp,
      scale);

  return std::make_tuple(
      std::move(attention),
      std::move(log_sumexp),
      std::move(seed),
      std::move(offset));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl__efficient_attention_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_k,
    c10::optional<int64_t> max_seqlen_q,
    double dropout_p,
    int64_t custom_mask_type,
    bool compute_log_sumexp,
    c10::optional<double> scale,
    const c10::optional<at::Tensor>& causal_diagonal,
    const c10::optional<at::Tensor>& seqlen_k) {
  // ndim
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(2);
  const int64_t max_seqlen_batch_q = query.size(1);
  const int64_t embedding = query.size(3);
  const int64_t max_seqlen_batch_k = key.size(1);
  const int64_t embedding_value = value.size(-1);

  at::Tensor cumulative_sequence_length_q, cumulative_sequence_length_k;
  TORCH_CHECK(cu_seqlens_q.has_value() == cu_seqlens_k.has_value());
  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(
        cu_seqlens_q.value().dtype() == at::kInt,
        "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(
        cu_seqlens_q.value().device().is_privateuseone(),
        "cu_seqlens_q must be on MLU device");
    CHECK_SHAPE(cu_seqlens_q.value(), batch_size + 1);
    cumulative_sequence_length_q = cnnl_contiguous(cu_seqlens_q.value());
  } else {
    cumulative_sequence_length_q = at::arange(
        0,
        (batch_size + 1) * max_seqlen_batch_q,
        max_seqlen_batch_q,
        c10::TensorOptions().device(at::kPrivateUse1).dtype(at::kInt));
  }
  if (cu_seqlens_k.has_value()) {
    TORCH_CHECK(
        cu_seqlens_k.value().dtype() == at::kInt,
        "cu_seqlens_k must have dtype int32");
    TORCH_CHECK(
        cu_seqlens_k.value().device().is_privateuseone(),
        "cu_seqlens_k must be on MLU device");
    CHECK_SHAPE(cu_seqlens_k.value(), batch_size + 1);
    cumulative_sequence_length_k = cnnl_contiguous(cu_seqlens_k.value());
  } else {
    cumulative_sequence_length_k = at::arange(
        0,
        (batch_size + 1) * max_seqlen_batch_k,
        max_seqlen_batch_k,
        c10::TensorOptions().device(at::kPrivateUse1).dtype(at::kInt));
  }

  int64_t Nnz_q{batch_size * max_seqlen_batch_q};
  int64_t Nnz_kv{batch_size * max_seqlen_batch_k};

  at::Tensor q_t_c = cnnl_contiguous(query);
  at::Tensor query_reshaped = q_t_c.view({Nnz_q, num_heads, embedding});
  at::Tensor k_t_c = cnnl_contiguous(key);
  at::Tensor key_reshaped = k_t_c.view({Nnz_kv, num_heads, embedding});
  at::Tensor v_t_c = cnnl_contiguous(value);
  at::Tensor value_reshaped = v_t_c.view({Nnz_kv, num_heads, embedding_value});
  c10::optional<Tensor> bias_c;
  if (bias.has_value()) {
    bias_c = cnnl_contiguous(bias.value());
  }
  const auto softmax_scale =
      sdp::calculate_scale(query_reshaped, scale).as_float_unchecked();

  at::Tensor output = at::empty(
      {batch_size * max_seqlen_batch_q, num_heads, embedding_value},
      query.options());

  auto [log_sumexp, seed, offset] = cnnl_mem_eff_fwd_internal(
      query_reshaped,
      key_reshaped,
      value_reshaped,
      bias_c,
      output,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      custom_mask_type,
      compute_log_sumexp,
      softmax_scale);

  output =
      output.view({batch_size, max_seqlen_batch_q, num_heads, embedding_value})
          .transpose(1, 2);
  return std::make_tuple(output, log_sumexp, seed, offset);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl__scaled_dot_product_efficient_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    double dropout_p,
    std::array<bool, 4> grad_input_mask,
    bool is_causal,
    c10::optional<double> scale) {
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }

  auto grad_out = grad_out_.transpose(1, 2);
  auto out_t = out.transpose(1, 2);
  auto q_t = query.transpose(1, 2);
  auto k_t = key.transpose(1, 2);
  auto v_t = value.transpose(1, 2);

  at::Tensor grad_q, grad_k, grad_v, grad_bias;

  // This is needed because SaveVarible automatically converts
  // c10::optional to undefined tensor
  c10::optional<Tensor> kernel_bias;
  if (attn_bias.defined()) {
    kernel_bias = attn_bias;
  }

  int64_t max_seqlen_q = q_t.size(1);
  int64_t max_seqlen_k = k_t.size(1);

  sdp::CustomMaskType custom_mask_type = is_causal
      ? sdp::CustomMaskType::CausalFromTopLeft
      : sdp::CustomMaskType::NoCustomMask;

  std::tie(grad_q, grad_k, grad_v, grad_bias) =
      at::_efficient_attention_backward(
          grad_out,
          q_t,
          k_t,
          v_t,
          kernel_bias,
          out_t,
          c10::nullopt,
          c10::nullopt,
          max_seqlen_k,
          max_seqlen_q,
          logsumexp,
          dropout_p,
          philox_seed,
          philox_offset,
          static_cast<int64_t>(custom_mask_type),
          grad_input_mask[3],
          scale,
          c10::nullopt); // num_split_keys

  return std::make_tuple(
      grad_q.transpose(1, 2),
      grad_k.transpose(1, 2),
      grad_v.transpose(1, 2),
      grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
cnnl__efficient_attention_backward(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& out,
    const c10::optional<at::Tensor>& cu_seqlens_q,
    const c10::optional<at::Tensor>& cu_seqlens_k,
    int64_t max_seqlen_k,
    int64_t max_seqlen_q,
    const at::Tensor& logsumexp,
    double dropout_p,
    const at::Tensor& philox_seed,
    const at::Tensor& philox_offset,
    int64_t custom_mask_type,
    bool bias_requires_grad,
    c10::optional<double> scale,
    c10::optional<int64_t> num_splits_key) {
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  // ndim
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 4);

  // batch size
  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // seqlen
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));
  TORCH_CHECK(value.size(3) == grad_out_.size(3));

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(2);
  const int64_t embedding = query.size(3);
  const int64_t embedding_value = value.size(3);

  int64_t Nnz_q{batch_size * max_seqlen_q};
  int64_t Nnz_kv{batch_size * max_seqlen_k};

  auto grad_out_c = cnnl_contiguous(grad_out_);
  auto grad_out_reshaped =
      grad_out_c.reshape({{Nnz_q, num_heads, embedding_value}});
  auto out_c = cnnl_contiguous(out);
  auto out_reshaped = out_c.reshape({Nnz_q, num_heads, embedding_value});
  auto q_c = cnnl_contiguous(query);
  at::Tensor query_reshaped = q_c.view({Nnz_q, num_heads, embedding});
  auto k_c = cnnl_contiguous(key);
  at::Tensor key_reshaped = k_c.view({Nnz_kv, num_heads, embedding});
  auto v_c = cnnl_contiguous(value);
  at::Tensor value_reshaped = v_c.view({Nnz_kv, num_heads, embedding_value});
  c10::optional<Tensor> bias_c;
  if (bias.has_value()) {
    bias_c = cnnl_contiguous(bias.value());
  }
  at::Tensor grad_q = at::empty_like(query_reshaped);
  at::Tensor grad_k = at::empty_like(key_reshaped);
  at::Tensor grad_v = at::empty_like(value_reshaped);
  at::Tensor grad_bias;
  if (bias_requires_grad && bias.has_value()) {
    grad_bias = at::empty_like(*bias);
  } else {
    grad_bias = at::empty(
        {batch_size * num_heads, max_seqlen_q, max_seqlen_k}, query.options());
  }

  at::Tensor cumulative_sequence_length_q, cumulative_sequence_length_k;
  TORCH_CHECK(cu_seqlens_q.has_value() == cu_seqlens_k.has_value());
  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(
        cu_seqlens_q.value().dtype() == at::kInt,
        "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(
        cu_seqlens_q.value().device().is_privateuseone(),
        "cu_seqlens_q must be on MLU device");
    CHECK_SHAPE(cu_seqlens_q.value(), batch_size + 1);
    cumulative_sequence_length_q = cnnl_contiguous(cu_seqlens_q.value());
  } else {
    cumulative_sequence_length_q = at::arange(
        0,
        (batch_size + 1) * max_seqlen_q,
        max_seqlen_q,
        c10::TensorOptions().device(at::kPrivateUse1).dtype(at::kInt));
  }
  if (cu_seqlens_k.has_value()) {
    TORCH_CHECK(
        cu_seqlens_k.value().dtype() == at::kInt,
        "cu_seqlens_k must have dtype int32");
    TORCH_CHECK(
        cu_seqlens_k.value().device().is_privateuseone(),
        "cu_seqlens_k must be on MLU device");
    CHECK_SHAPE(cu_seqlens_k.value(), batch_size + 1);
    cumulative_sequence_length_k = cnnl_contiguous(cu_seqlens_k.value());
  } else {
    cumulative_sequence_length_k = at::arange(
        0,
        (batch_size + 1) * max_seqlen_k,
        max_seqlen_k,
        c10::TensorOptions().device(at::kPrivateUse1).dtype(at::kInt));
  }

  const auto softmax_scale =
      sdp::calculate_scale(query_reshaped, scale).as_float_unchecked();

  std::tie(grad_q, grad_k, grad_v) = cnnl_mem_eff_bwd_internal(
      grad_out_reshaped,
      query_reshaped,
      key_reshaped,
      value_reshaped,
      bias_c,
      out_reshaped,
      logsumexp,
      grad_q,
      grad_k,
      grad_v,
      grad_bias,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_q,
      max_seqlen_k,
      dropout_p,
      softmax_scale,
      false, /*zero_tensors = false for all calls here*/
      philox_seed,
      philox_offset,
      custom_mask_type);

  grad_q = grad_q.view({batch_size, max_seqlen_q, num_heads, embedding});
  grad_k = grad_k.view({batch_size, max_seqlen_k, num_heads, embedding});
  grad_v = grad_v.view({batch_size, max_seqlen_k, num_heads, embedding_value});
  grad_bias =
      grad_bias.view({batch_size, num_heads, max_seqlen_q, max_seqlen_k});
  return std::make_tuple(grad_q, grad_k, grad_v, grad_bias);
}

} // namespace ops
} // namespace torch_mlu