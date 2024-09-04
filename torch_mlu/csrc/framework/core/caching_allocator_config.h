#pragma once

#include <c10/util/Exception.h>
#include <c10/util/llvmMathExtras.h>

#include <atomic>
#include <vector>

#include "framework/core/caching_allocator.h"

namespace torch_mlu::MLUCachingAllocator {

class MLUAllocatorConfig {
 public:
  static size_t max_split_size() {
    return instance().m_max_split_size;
  }
  static double garbage_collection_threshold() {
    return instance().m_garbage_collection_threshold;
  }

  static bool expandable_segments() {
    if (instance().m_expandable_segments) {
      TORCH_WARN_ONCE("expandable_segments currently not supported on MLU.")
    }
    return false;
    // return instance().m_expandable_segments;
  }

  static bool use_linear_memory() {
    return instance().m_use_linear_memory;
  }

  // This is used to round-up allocation size to nearest power of 2 divisions.
  // More description below in function roundup_power2_next_division
  // As an example, if we want 4 divisions between 2's power, this can be done
  // using env variable: PYTORCH_MLU_ALLOC_CONF=roundup_power2_divisions:4
  static size_t roundup_power2_divisions(size_t size);

  static std::vector<size_t> roundup_power2_divisions() {
    return instance().m_roundup_power2_divisions;
  }

  static std::string last_allocator_settings() {
    std::lock_guard<std::mutex> lock(
        instance().m_last_allocator_settings_mutex);
    return instance().m_last_allocator_settings;
  }

  static MLUAllocatorConfig& instance();

  void parseArgs(const char* env);

 private:
  MLUAllocatorConfig();

  void lexArgs(const char* env, std::vector<std::string>& config);
  void consumeToken(
      const std::vector<std::string>& config,
      size_t i,
      const char c);
  size_t parseMaxSplitSize(const std::vector<std::string>& config, size_t i);
  size_t parseGarbageCollectionThreshold(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseRoundUpPower2Divisions(
      const std::vector<std::string>& config,
      size_t i);
  size_t parseAllocatorConfig(
      const std::vector<std::string>& config,
      size_t i,
      bool& used_cnrtMallocAsync);

  std::atomic<size_t> m_max_split_size;
  std::vector<size_t> m_roundup_power2_divisions;
  std::atomic<double> m_garbage_collection_threshold;
  std::atomic<bool> m_expandable_segments;
  std::atomic<bool> m_use_linear_memory;
  std::string m_last_allocator_settings;
  std::mutex m_last_allocator_settings_mutex;
};

// General caching allocator utilities
TORCH_MLU_API void setAllocatorSettings(const std::string& env);

} // namespace torch_mlu::MLUCachingAllocator
