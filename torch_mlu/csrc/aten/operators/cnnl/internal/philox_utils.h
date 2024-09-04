#pragma once
#include <ATen/ATen.h>

namespace torch_mlu {
namespace ops {

inline int64_t calc_counter_offset(int64_t nelem, int64_t thread_num) {
  const int64_t UNROLL = 4;
  return ((nelem - 1) / (thread_num * UNROLL) + 1) * UNROLL;
}

} // namespace ops
} // namespace torch_mlu
