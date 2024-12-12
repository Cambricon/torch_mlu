#pragma once
#include <cmath>
#include "ATen/Context.h"
#include "ATen/core/Tensor.h"
#include "ATen/core/grad_mode.h"
#include "ATen/NestedTensorImpl.h"
#include "c10/util/Array.h"
#include "framework/core/device.h"
#include "utils/Export.h"

namespace torch_mlu {
namespace sdp {

constexpr int32_t num_backends = 3;

enum class SDPBackend {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2
};

struct sdp_params {
  at::Tensor query;
  at::Tensor key;
  at::Tensor value;
  std::optional<at::Tensor> attn_mask;
  double dropout;
  bool is_causal;
};

enum class CustomMaskType {
  NoCustomMask = 0,
  CausalFromTopLeft = 1,
  CausalFromBottomRight = 2,
  NumCustomMaskTypes,
};

SDPBackend select_sdp_backend(const sdp_params& kernel_params);
TORCH_MLU_API bool can_use_flash_attention(
    const sdp_params& params,
    bool debug);
TORCH_MLU_API bool can_use_mem_efficient_attention(
    const sdp_params& params,
    bool debug);

inline c10::SymFloat calculate_scale(
    const at::Tensor& query,
    std::optional<double> scale) {
  const auto softmax_scale = scale.has_value()
      ? scale.value()
      : (c10::SymFloat(1.0) / (c10::SymFloat(query.sym_size(-1)).sqrt()));
  return c10::SymFloat(softmax_scale);
}

using c10::array_of;

inline bool input_requires_grad(sdp_params params) {
  const bool any_inputs_require_grad = params.query.requires_grad() ||
      params.key.requires_grad() || params.value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  return any_inputs_require_grad && gradmode_enabled;
}

inline bool has_for_nested_inputs(const sdp_params& params) {
  return (params.query.is_nested() && params.query.layout() == c10::kStrided) ||
      (params.key.is_nested() && params.key.layout() == c10::kStrided) ||
      (params.value.is_nested() && params.value.layout() == c10::kStrided);
}

inline bool has_only_dense_inputs(const sdp_params& params) {
  return !params.query.is_nested() && !params.key.is_nested() &&
      !params.value.is_nested();
}

inline bool try_broadcast_param_size(
    const c10::SymInt q_size,
    const c10::SymInt k_size,
    const c10::SymInt v_size,
    c10::string_view param_name,
    bool debug) {
  auto max_size = std::max({q_size, k_size, v_size});
  if ((q_size != max_size && q_size != 1) ||
      (k_size != max_size && k_size != 1) ||
      (v_size != max_size && v_size != 1)) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels require query, key and value to have broadcastable ",
          param_name,
          "got Query ",
          param_name,
          q_size,
          ", Key ",
          param_name,
          k_size,
          ", Value ",
          param_name,
          v_size,
          " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_runtime_disabled_flash(const sdp_params& params, bool debug) {
  // We check the global context to see if user has explicitly turned of flash
  // sdp kernels
  if (!at::globalContext().userEnabledFlashSDP()) {
    if (debug) {
      TORCH_WARN("Flash attention has been runtime disabled.");
    }
    return false;
  }
  return true;
}

inline bool check_runtime_disabled_mem_efficient(
    const sdp_params& params,
    bool debug) {
  // We check the global context to see if user has explicitly turned of
  // mem_efficient sdp kernels
  if (!at::globalContext().userEnabledMemEfficientSDP()) {
    if (debug) {
      TORCH_WARN("Memory Efficient attention has been runtime disabled.");
    }
    return false;
  }
  return true;
}

inline bool check_tensor_shapes(const sdp_params& params, bool debug) {
  auto query_dim = params.query.dim();
  if (!(query_dim == params.key.dim() && query_dim == params.value.dim() &&
        (query_dim == 4))) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels requires query, key and value to be 4 dimensional, but got Query dim: ",
          query_dim,
          ", Key dim: ",
          params.key.dim(),
          ", Value dim: ",
          params.value.dim(),
          " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_safe_kv_broadcast(at::Tensor param, bool debug) {
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  auto seq_len = nt_tensor_impl->opt_size(2);
  if (!seq_len.has_value()) {
    if (debug) {
      TORCH_WARN(
          "For both fused kernels, if one of key/value batch_size requires "
          "broadcasting and the other does not, then the other must have a ",
          "consistent seq_len dim.")
    }
    return false;
  }
  return true;
}

inline bool check_for_attn_mask(const sdp_params& params, bool debug) {
  if (params.attn_mask.has_value()) {
    if (debug) {
      TORCH_WARN("Flash Attention does not support non-null attn_mask.");
    }
    return false;
  }
  return true;
}

inline bool check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
    at::Tensor param,
    c10::string_view param_name,
    bool debug) {
  const auto nt_tensor_impl = at::native::get_nested_tensor_impl(param);
  const at::Tensor& sizes = nt_tensor_impl->get_nested_sizes();
  auto num_head_dims = nt_tensor_impl->opt_size(1);
  if (!num_head_dims.has_value()) {
    // num_head_dims is ragged
    if (debug) {
      TORCH_WARN(
          "Fused kernels do not support ragged num_head_dims, ",
          param_name,
          "has a ragged num_heads.");
    }
    return false;
  }

  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = param.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] == 0) {
      if (debug) {
        TORCH_WARN(
            "Fused kernels do not support seq_len == 0, ",
            param_name,
            "has a seq len of 0.");
      }
      return false;
    }
  }
  return true;
}

inline bool check_for_seq_len_0_nested_tensor(
    const sdp_params& params,
    bool debug) {
  // When this function is called we are assured that the nt is dim==4
  bool q_is_safe = params.query.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.query, "query ", debug)
      : true;
  // short circuit if any is unsafe
  if (!q_is_safe) {
    return false;
  }

  bool k_is_safe = params.key.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.key, "key ", debug)
      : true;
  if (!k_is_safe) {
    return false;
  }

  bool v_is_safe = params.value.is_nested()
      ? check_for_seq_len_0_and_consistent_head_dim_nested_tensor_helper(
            params.value, "value ", debug)
      : true;
  if (!v_is_safe) {
    return false;
  }

  // We now know none of the inputs have ragged num_heads, so we can safely
  // access .size(1)
  auto q_num_heads = params.query.size(1);
  auto k_num_heads = params.key.size(1);
  auto v_num_heads = params.value.size(1);
  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  if (!same_num_heads) {
    if (input_requires_grad(params)) {
      if (debug) {
        TORCH_WARN(
            "Both fused kernels do not support training with broadcasted NT inputs.");
      }
      return false;
    }
    return try_broadcast_param_size(
        q_num_heads, k_num_heads, v_num_heads, "num heads ", debug);
  }

  return true;
}

template <typename dtype_vector>
inline bool check_tensor_dtype(
    sdp_params params,
    dtype_vector allowed_dtypes,
    bool debug) {
  auto query_dtype = params.query.dtype();
  if (!(query_dtype == params.key.dtype() &&
        query_dtype == params.value.dtype() &&
        (std::find(allowed_dtypes.begin(), allowed_dtypes.end(), query_dtype) !=
         allowed_dtypes.end()))) {
    if (debug) {
      TORCH_WARN(
          "Expected query, key and value to all be of dtype: {",
          c10::Join(", ", allowed_dtypes),
          "}. Got ",
          "Query dtype: ",
          params.query.dtype(),
          ", Key dtype: ",
          params.key.dtype(),
          ", and Value dtype: ",
          params.value.dtype(),
          " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_requires_grad_and_nested(
    const sdp_params& params,
    bool debug) {
  if (input_requires_grad(params)) {
    if (debug) {
      TORCH_WARN(
          "Memory efficient attention currently doesn't support training with NT inputs.");
    }
    return false;
  }
  return true;
}

inline bool check_all_tensors_on_device(const sdp_params& params, bool debug) {
  if (params.query.device().type() != at::DeviceType::PrivateUse1) {
    if (debug) {
      TORCH_WARN(
          "All tensors need to be on mlu device. Got query on device: ",
          params.query.device(),
          ", key on device: ",
          params.key.device(),
          ", value on device: ",
          params.value.device());
    }
    return false;
  }
  return true;
}

inline bool check_batch_size_and_num_heads_dense(
    const sdp_params& params,
    bool debug) {
  auto q_batch_size = params.query.sym_size(0);
  auto k_batch_size = params.key.sym_size(0);
  auto v_batch_size = params.value.sym_size(0);

  bool same_batch_size =
      q_batch_size == k_batch_size && q_batch_size == v_batch_size;

  auto q_num_heads = params.query.sym_size(1);
  auto k_num_heads = params.key.sym_size(1);
  auto v_num_heads = params.value.sym_size(1);
  bool same_num_heads =
      q_num_heads == k_num_heads && q_num_heads == v_num_heads;

  if (!(same_batch_size && same_num_heads)) {
    if (debug) {
      TORCH_WARN(
          "For dense inputs, both fused kernels require query, key and value to have the same batch_size and num_heads. ",
          "Query.sizes(): ",
          params.query.sizes(),
          ", Key sizes(): ",
          params.key.sizes(),
          ", Value sizes(): ",
          params.value.sizes(),
          " instead. To broadcast dense inputs, try using unsqueeze and expand_to before passing them into the kernel.");
    }
    return false;
  }
  return true;
}

inline bool check_nonzero_sequence_lengths_dense(
    const sdp_params& params,
    bool debug) {
  bool zero_seq_len_q = params.query.sym_size(-2) == 0;
  bool zero_seq_len_k = params.key.sym_size(-2) == 0;
  if (zero_seq_len_q || zero_seq_len_k) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels do not support zero seq_len_q or seq_len_kv.");
    }
    return false;
  }
  return true;
}

template <bool ignore_singleton_dim>
inline bool check_last_dim_stride_equals_1_dense(
    sdp_params const& params,
    bool debug) {
  bool qkv_strides_equal_1 = params.query.sym_stride(-1) == 1 &&
      params.key.sym_stride(-1) == 1 && params.value.sym_stride(-1) == 1;

  if (ignore_singleton_dim) {
    qkv_strides_equal_1 = qkv_strides_equal_1 || params.query.sym_size(-1) == 1;
  }
  bool mask_stride_equal_1 = params.attn_mask.has_value()
      ? params.attn_mask.value().sym_stride(-1) == 1
      : true;
  if (!(qkv_strides_equal_1 && mask_stride_equal_1)) {
    if (debug) {
      std::ostringstream epilogue_message;
      if (params.attn_mask.has_value()) {
        epilogue_message << ", Attn_mask.stride(-1): "
                         << params.attn_mask.value().sym_stride(-1);
      }
      epilogue_message << " instead.";
      TORCH_WARN(
          "Both fused kernels require the last dimension of the input to have stride 1. ",
          "Got Query.stride(-1): ",
          params.query.sym_stride(-1),
          ", Key.stride(-1): ",
          params.key.sym_stride(-1),
          ", Value.stride(-1): ",
          params.value.sym_stride(-1),
          epilogue_message.str());
    }

    return false;
  }
  return true;
}

inline bool check_batch_size_nested(sdp_params const& params, bool debug) {
  // This is expected to be called after check_tensor_shapes ensuring that the
  // size() calls won't error since the inputs are all 4 dimensional
  auto q_batch_size = params.query.sym_size(0);
  auto k_batch_size = params.key.sym_size(0);
  auto v_batch_size = params.value.sym_size(0);

  bool same_batch_size =
      q_batch_size == k_batch_size && q_batch_size == v_batch_size;

  // num_heads logic for nested input is checked in
  // check_for_seq_len_0_nested_tensor as there is handling there to make sure
  // num_heads is not ragged
  bool broadcastable_batch_size = true;
  if (!same_batch_size) {
    if (input_requires_grad(params)) {
      if (debug) {
        TORCH_WARN(
            "Both fused kernels do not support training with broadcasted NT inputs.");
      }
      return false;
    }
    // try to broadcast batchsize
    broadcastable_batch_size = try_broadcast_param_size(
        q_batch_size, k_batch_size, v_batch_size, "batch size ", debug);

    // if only one of k or v require broadcasting of batch size, the other
    // must have a consistent seq_len dim
    if (broadcastable_batch_size) {
      if (k_batch_size == 1 && v_batch_size != 1 &&
          !check_safe_kv_broadcast(params.value, debug)) {
        return false;
      }
      if (v_batch_size == 1 && k_batch_size != 1 &&
          !check_safe_kv_broadcast(params.key, debug)) {
        return false;
      }
    }
  }
  return broadcastable_batch_size;
}

inline bool check_head_dim_size_flash_nested(
    sdp_params const& params,
    bool debug) {
  const auto max_size = c10::SymInt(256);
  const auto query_size_last = params.query.sym_size(-1);
  const auto key_size_last = params.key.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  bool same_head_dim_size =
      query_size_last == key_size_last && query_size_last == value_size_last;
  if (!(same_head_dim_size && (query_size_last % 8 == 0) &&
        (query_size_last <= max_size))) {
    if (debug) {
      TORCH_WARN(
          "For NestedTensor inputs, Flash attention requires q,k,v to have the same last dimension and to be a multiple of 8 and less than or equal to 256.",
          " Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          params.value.sym_size(-1),
          " instead.");
    }
    return false;
  }
  return true;
}

inline bool check_flash_causal_non_square_seqlens(
    const sdp_params& params,
    bool debug) {
  if (params.is_causal && !params.query.is_nested() &&
      !params.key.is_nested() &&
      params.query.sym_size(-2) != params.key.sym_size(-2)) {
    if (debug) {
      TORCH_WARN(
          "Flash attention does not support the is_causal flag when seqlen_q != seqlen_k. ",
          "Got seqlen_q: ",
          params.query.sym_size(-2),
          " seqlen_k: ",
          params.key.sym_size(-2),
          ". If you would like to use causal attention with non-square masks, please see CausalAttnMask.");
    }
    return false;
  }
  return true;
}

inline bool check_head_dim_size(sdp_params const& params, bool debug) {
  const auto query_size_last = params.query.sym_size(-1);
  const auto value_size_last = params.value.sym_size(-1);
  if (!(query_size_last == params.key.sym_size(-1) &&
        (c10::SymInt(256) >= query_size_last) &&
        (query_size_last >= c10::SymInt(1)) &&
        (c10::SymInt(256) >= value_size_last) &&
        (value_size_last >= c10::SymInt(1)))) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels require query.size(-1)==key.size(-1),",
          "256>=query.size(-1)>=1 and 256>=value.size(-1)>=1.",
          "Got Query.size(-1): ",
          query_size_last,
          ", Key.size(-1): ",
          params.key.sym_size(-1),
          ", Value.size(-1): ",
          value_size_last);
    }
    return false;
  }
  return true;
}

inline bool check_for_seq_len_1_nested_tensor(sdp_params params, bool debug) {
  // When this function is called we are assured that the nt is dim==4
  if (!params.query.is_nested()) {
    return true;
  }

  const auto nt_q_tensor_impl =
      at::native::get_nested_tensor_impl(params.query);
  const at::Tensor& sizes = nt_q_tensor_impl->get_nested_sizes();
  auto* sizes_ptr = sizes.data_ptr<int64_t>();
  const int64_t n_tensors = params.query.size(0);
  const int64_t size_tensor_stride = sizes.stride(0);

  // This is being called inside sdp with shape [batch, heads, {seq_len}, dim]
  for (const auto i : c10::irange(n_tensors)) {
    if (sizes_ptr[(i * size_tensor_stride) + 1] <= 1) {
      if (debug) {
        TORCH_WARN(
            "Packed projection for fused kernels does not support sequence_length <= 1");
      }
      return false;
    }
  }

  return true;
}

inline double ceiling(double number, double significance) {
  return ceil(number / significance) * significance;
}

inline bool check_fused_kernel_mlu_support(
    sdp_params const& params,
    bool debug) {
  DeviceProp* prop = torch_mlu::getDeviceProperties(params.query.get_device());
  if ((*prop).major != 5) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels only supports specified MLU series.",
          "Attempting to run on ",
          (*prop).name,
          ".");
    }
    return false;
  }

  auto batch_size = params.query.size(0);
  auto num_heads = params.query.size(1);
  auto seq_len_q = params.query.size(-2);
  auto seq_len_k = params.key.size(-2);
  auto head_size = params.query.size(-1);

  auto split_cond = batch_size * num_heads * ceil(((float)seq_len_k) / 256);
  if (!(split_cond > 12)) {
    if (debug) {
      TORCH_WARN(
          "Both fused kernels require split condition greater than 12.",
          "Got ",
          split_cond,
          ".");
    }
    return false;
  }

  auto valid_data_ratio_cond = seq_len_q * seq_len_k /
      (ceiling(seq_len_q, 512) * ceiling(seq_len_k, 256));
  bool valid_cond_1 =
      (seq_len_q * head_size >= 5190) && (seq_len_k * head_size >= 5190);
  bool valid_cond_2 = (seq_len_q >= 1024) && (seq_len_k * head_size >= 5190);
  bool valid_cond_3 = !params.attn_mask.has_value();
  if (!(((valid_cond_1 || valid_cond_2) && valid_cond_3) ||
        (valid_data_ratio_cond >= 0.6))) {
    if (debug) {
      TORCH_WARN("The valid data required by the Fused kernel is not met.");
    }
    return false;
  }

  return true;
}

} // namespace sdp
} // namespace torch_mlu
