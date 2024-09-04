#include <gtest/gtest.h>

#include "framework/core/caching_allocator_config.h"
#include "utils/common.h"

namespace torch_mlu {

TEST(EnvTest, PytorchNoMluMemoryCachingTest) {
  setenv("PYTORCH_NO_CUDA_MEMORY_CACHING", "1", 1);
  static const std::string env_str = []() {
    const std::vector<std::string> PYTORCH_NO_MLU_MEMORY_CACHING = {
        "PYTORCH_NO_MLU_MEMORY_CACHING", "PYTORCH_NO_CUDA_MEMORY_CACHING"};
    return getCvarString(PYTORCH_NO_MLU_MEMORY_CACHING, "");
  }();
  EXPECT_EQ(env_str, "1");
}

TEST(EnvTest, PytorchMluAllocConfTest) {
  setenv("PYTORCH_CUDA_ALLOC_CONF", "1", 1);
  const std::vector<std::string> PYTORCH_MLU_ALLOC_CONF = {
      "PYTORCH_MLU_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"};
  auto env_str = getCvarString(PYTORCH_MLU_ALLOC_CONF, "");
  EXPECT_EQ(env_str, "1");
}

TEST(EnvTest, TorchAllowTf32CnmatmulOverrideTest) {
  setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1", 1);
  static const bool env_bool = []() {
    const std::vector<std::string> TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE = {
        "TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"};
    return getCvarBool(TORCH_ALLOW_TF32_CNMATMUL_OVERRIDE, false);
  }();
  EXPECT_TRUE(env_bool);
}

} // namespace torch_mlu
