#include "sdp_utils.h"
#include "ATen/core/grad_mode.h"
#include "c10/core/ScalarType.h"

namespace torch_mlu {
namespace sdp {

std::array<SDPBackend, num_backends> priority_order(sdp_params params) {
  constexpr std::array<SDPBackend, num_backends> default_order{
      SDPBackend::flash_attention,
      SDPBackend::efficient_attention,
      SDPBackend::math};

  constexpr std::array<SDPBackend, num_backends> efficient_first{
      SDPBackend::efficient_attention,
      SDPBackend::flash_attention,
      SDPBackend::math};
  // Logic is taken from xformers
  // FlashAttention parallelizes across "batch_size * num_heads"
  // MemEff parallelizes across "batch_size * num_heads * num_queries" and can
  // be more efficient. batch_size, q_len, num_heads, k = inp.query.shape
  if (has_for_nested_inputs(params)) {
    return efficient_first;
  }
  if (params.query.dim() != 4) {
    return default_order;
  }
  const auto batch_size{params.query.sym_size(0)},
      num_heads{params.query.sym_size(1)},
      query_lengths{params.query.sym_size(2)},
      head_dim{params.query.sym_size(3)};
  if (batch_size > 0) {
    const auto threads_flash = batch_size * num_heads;
    const auto threads_cutlass =
        threads_flash * (query_lengths / c10::SymInt(64));
    bool more_threads_cutlass = (threads_cutlass / 2) >= threads_flash;
    bool small_threads_flash = threads_flash < 60;
    bool large_head_dim = head_dim.max(params.key.sym_size(3)) == 128;

    // The training heuristic is taken from
    // https://github.com/pytorch/pytorch/pull/99644 Revisit when updated
    // cutlass kernel is upstreamed.
    if (input_requires_grad(params)) {
      if (6 * threads_flash > query_lengths)
        return efficient_first;
    } else if ((small_threads_flash && more_threads_cutlass) || large_head_dim)
      return efficient_first;
  }
  return default_order;
}

bool use_flash_attention(sdp_params params, bool debug) {
  constexpr auto constraints = array_of<bool (*)(const sdp_params&, bool)>(
      check_runtime_disabled_flash,
      check_tensor_shapes, // (bach_size, num_heads, query_length, head_dim)
      check_batch_size_and_num_heads,
      check_for_attn_mask,
      check_head_dim_size,
      check_for_seq_len_0_nested_tensor,
      check_nonzero_sequence_lengths,
      check_last_dim_stride_equals_1,
      check_fused_kernel_mlu_support);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
    }
  }

  constexpr auto supported_flash_dtypes =
      array_of<at::ScalarType>(at::kHalf, at::kBFloat16);
  return check_tensor_dtype(params, supported_flash_dtypes, debug);
}

bool use_mem_efficient_attention(sdp_params params, bool debug) {
  // Constraints specific to mem efficient attention
  constexpr auto constraints = array_of<bool (*)(const sdp_params&, bool)>(
      check_runtime_disabled_mem_efficient,
      check_requires_grad_and_nested,
      check_tensor_shapes,
      check_batch_size_and_num_heads,
      check_head_dim_size,
      check_for_seq_len_0_nested_tensor,
      check_nonzero_sequence_lengths,
      check_last_dim_stride_equals_1,
      check_fused_kernel_mlu_support);
  for (auto& constraint : constraints) {
    if (!constraint(params, debug)) {
      return false;
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
  // Get ideal kernel ordering
  const auto ordering = priority_order(kernel_params);

  // Because TORCHCHECK checks if condition is true we negate debug so that
  // The statements will be printed when debug is true
  bool print_debug = false;
  for (auto& backend : ordering) {
    switch (backend) {
      case SDPBackend::flash_attention:
        if (use_flash_attention(kernel_params, print_debug)) {
          return SDPBackend::flash_attention;
        }
        break;
      case SDPBackend::efficient_attention:
        if (use_mem_efficient_attention(kernel_params, print_debug)) {
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
  use_mem_efficient_attention(kernel_params, print_debug);
  TORCH_WARN("Flash attention kernel not used because:");
  use_flash_attention(kernel_params, print_debug);
  TORCH_CHECK(!print_debug, "No available kernel.  Aborting execution.")
  return SDPBackend::error;
}

} // namespace sdp
} // namespace torch_mlu