#include "sdp_utils.h"
#include "ATen/core/grad_mode.h"
#include "c10/core/ScalarType.h"

namespace torch_mlu {
namespace sdp {

std::array<SDPBackend, num_backends> priority_order(const sdp_params& params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::flash_attention,
      SDPBackend::efficient_attention,
      SDPBackend::math};
  return default_order;
}

bool can_use_flash_attention(const sdp_params& params, bool debug) {
  constexpr auto general_constraints =
      array_of<bool (*)(sdp_params const&, bool)>(
          check_runtime_disabled_flash,
          check_all_tensors_on_device,
          check_tensor_shapes,
          check_for_attn_mask,
          check_head_dim_size,
          check_flash_causal_non_square_seqlens);
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints =
        array_of<bool (*)(sdp_params const&, bool)>(
            check_batch_size_nested,
            check_head_dim_size_flash_nested,
            check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = array_of<bool (*)(
        sdp_params const&, bool)>(
        check_batch_size_and_num_heads_dense,
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<true /*ignore_singleton_dim=*/>,
        check_fused_kernel_mlu_support);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  constexpr auto supported_flash_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
  return check_tensor_dtype(params, supported_flash_dtypes, debug);
}

bool can_use_mem_efficient_attention(const sdp_params& params, bool debug) {
  constexpr auto general_constraints =
      array_of<bool (*)(sdp_params const&, bool)>(
          check_runtime_disabled_mem_efficient,
          check_all_tensors_on_device,
          check_tensor_shapes,
          check_head_dim_size);
  for (auto& constraint : general_constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  if (has_for_nested_inputs(params)) {
    constexpr auto nested_constraints =
        array_of<bool (*)(sdp_params const&, bool)>(
            check_requires_grad_and_nested,
            check_batch_size_nested,
            check_for_seq_len_0_nested_tensor);
    for (auto& constraint : nested_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }
  if (has_only_dense_inputs(params)) {
    constexpr auto dense_constraints = array_of<bool (*)(
        sdp_params const&, bool)>(
        check_batch_size_and_num_heads_dense,
        check_nonzero_sequence_lengths_dense,
        check_last_dim_stride_equals_1_dense<false /*ignore_singleton_dim=*/>,
        check_fused_kernel_mlu_support);
    for (auto& constraint : dense_constraints) {
      if (!constraint(params, debug)) {
        return false;
      }
    }
  }

  constexpr auto default_mem_efficient_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
  return check_tensor_dtype(params, default_mem_efficient_dtypes, debug);
}

SDPBackend select_sdp_backend(const sdp_params& kernel_params) {
  // This function defines the priority order of the different sdp backends
  // 1. Flash Attention2
  // 2. Mem Efficient Attention
  // 3. Math fallback
  auto& ctx = at::globalContext();
  if (!ctx.userEnabledMathSDP() && !ctx.userEnabledFlashSDP() &&
      !ctx.userEnabledMemEfficientSDP()) {
    return SDPBackend::error;
  }
  const auto ordering = priority_order(kernel_params);

  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::flash_attention:
        if (can_use_flash_attention(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::efficient_attention:
        if (can_use_mem_efficient_attention(kernel_params, print_debug)) {
          return SDPBackend::efficient_attention;
        }
        break;
      case SDPBackend::math:
        if (ctx.userEnabledMathSDP()) {
          return SDPBackend::math;
        }
        break;
      default:
        TORCH_CHECK(false, "Invalid backend");
    }
  }
  // If we have gotten to this point then two things have happened:
  // 1. use_flash_attention or use_mem_efficient did not satisfy the
  // constraints to be ran
  // 2. The user has explicitly disabled the math kernel
  // We then re-run the kernel checks with debug enabled to print out the
  // reason why the kernel was not selected
  print_debug = true;
  TORCH_WARN("Memory efficient kernel not used because:");
  can_use_mem_efficient_attention(kernel_params, print_debug);
  TORCH_WARN("Flash attention kernel not used because:");
  can_use_flash_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel.  Aborting execution.")
  return SDPBackend::error;
}

} // namespace sdp
} // namespace torch_mlu