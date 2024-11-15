#pragma once

#define MAX_TENSOR_NUM 31

#define ALIGN_UP(x, n) ((((x) + (n)-1) / (n)) * (n))
#define ALIGN_UP_DIV(x, n) (((x) + (n)-1) / (n))

namespace torch_mlu::ops {
static constexpr int reserve_stack_size = 32 * 1024;
} // namespace torch_mlu::ops

#ifdef __BANG_ARCH__
#define MAX_NRAM_SIZE \
  (__MLU_NRAM_SIZE__ * 1024 - torch_mlu::ops::reserve_stack_size)
#if __MLU_SRAM_SIZE__ == 0
#define MAX_SRAM_SIZE 0
#else
#define MAX_SRAM_SIZE \
  (__MLU_SRAM_SIZE__ * 1024 - torch_mlu::ops::reserve_stack_size)
#endif
#else
#define MAX_NRAM_SIZE (384 * 1024)
#define MAX_SRAM_SIZE (1920 * 1024)
#endif

typedef void* MLUaddr;

struct AddressList {
  void* addresses[MAX_TENSOR_NUM];
};

struct SizeList {
  int sizes[MAX_TENSOR_NUM];
};
