#include "framework/core/caching_allocator_config.h"
#include "framework/distributed/Utils.h"

namespace torch_mlu::MLUCachingAllocator {

constexpr size_t kRoundUpPowerOfTwoIntervals = 16;

MLUAllocatorConfig::MLUAllocatorConfig()
    : m_max_split_size(std::numeric_limits<size_t>::max()),
      m_garbage_collection_threshold(0),
      m_expandable_segments(false),
      m_release_lock_on_cnrtmalloc(false),
      m_use_linear_memory(false),
      m_last_allocator_settings("") {
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
}

size_t MLUAllocatorConfig::roundup_power2_divisions(size_t size) {
  size_t log_size = (63 - at::llvm::countLeadingZeros(size));

  // Our intervals start at 1MB and end at 64GB
  const size_t interval_start =
      63 - at::llvm::countLeadingZeros(static_cast<size_t>(1048576));
  const size_t interval_end =
      63 - at::llvm::countLeadingZeros(static_cast<size_t>(68719476736));
  TORCH_CHECK(
      (interval_end - interval_start == kRoundUpPowerOfTwoIntervals),
      "kRoundUpPowerOfTwoIntervals mismatch");

  int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

  index = std::max(0, index);
  index = std::min(index, static_cast<int>(kRoundUpPowerOfTwoIntervals) - 1);
  return instance().m_roundup_power2_divisions[index];
}

void MLUAllocatorConfig::lexArgs(
    const char* env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  size_t env_length = strlen(env);
  for (size_t i = 0; i < env_length; i++) {
    if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, env[i]);
    } else if (env[i] != ' ') {
      buf.emplace_back(static_cast<char>(env[i]));
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void MLUAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i].compare(std::string(1, c)) == 0,
      "Error parsing CachingAllocator settings, expected ",
      c,
      "");
}

size_t MLUAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  constexpr int mb = 1024 * 1024;
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > large_buffer_size_mlu / mb,
        "CachingAllocator option max_split_size_mb too small, must be > ",
        large_buffer_size_mlu / mb,
        "");
    val1 = std::max(val1, large_buffer_size_mlu / mb);
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / mb));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");
  }
  return i;
}

size_t MLUAllocatorConfig::parseGarbageCollectionThreshold(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  if (++i < config.size()) {
    double val1 = stod(config[i]);
    TORCH_CHECK(
        val1 > 0, "garbage_collection_threshold too small, set it 0.0~1.0", "");
    TORCH_CHECK(
        val1 < 1.0, "garbage_collection_threshold too big, set it 0.0~1.0", "");
    m_garbage_collection_threshold = val1;
  } else {
    TORCH_CHECK(
        false, "Error, expecting garbage_collection_threshold value", "");
  }
  return i;
}

size_t MLUAllocatorConfig::parseRoundUpPower2Divisions(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  bool first_value = true;

  if (++i < config.size()) {
    if (std::string_view(config[i]) == "[") {
      size_t last_index = 0;
      while (++i < config.size() && std::string_view(config[i]) != "]") {
        const std::string& val1 = config[i];
        size_t val2 = 0;

        consumeToken(config, ++i, ':');
        if (++i < config.size()) {
          val2 = stoi(config[i]);
        } else {
          TORCH_CHECK(
              false, "Error parsing roundup_power2_divisions value", "");
        }
        TORCH_CHECK(
            val2 == 0 || c10::llvm::isPowerOf2_64(val2),
            "For roundups, the divisons has to be power of 2 or 0 to disable roundup ",
            "");

        if (std::string_view(val1) == ">") {
          std::fill(
              std::next(
                  m_roundup_power2_divisions.begin(),
                  static_cast<std::vector<unsigned long>::difference_type>(
                      last_index)),
              m_roundup_power2_divisions.end(),
              val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(
              c10::llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ",
              "");

          size_t index = 63 - c10::llvm::countLeadingZeros(val1_long);
          index = std::max((size_t)0, index);
          index = std::min(index, m_roundup_power2_divisions.size() - 1);

          if (first_value) {
            std::fill(
                m_roundup_power2_divisions.begin(),
                std::next(
                    m_roundup_power2_divisions.begin(),
                    static_cast<std::vector<unsigned long>::difference_type>(
                        index)),
                val2);
            first_value = false;
          }
          if (index < m_roundup_power2_divisions.size()) {
            m_roundup_power2_divisions[index] = val2;
          }
          last_index = index;
        }

        if (std::string_view(config[i + 1]) != "]") {
          consumeToken(config, ++i, ',');
        }
      }
    } else {
      size_t val1 = stoi(config[i]);
      TORCH_CHECK(
          c10::llvm::isPowerOf2_64(val1),
          "For roundups, the divisons has to be power of 2 ",
          "");
      std::fill(
          m_roundup_power2_divisions.begin(),
          m_roundup_power2_divisions.end(),
          val1);
    }
  } else {
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value", "");
  }
  return i;
}

MLUAllocatorConfig& MLUAllocatorConfig::instance() {
  static MLUAllocatorConfig* s_instance = ([]() {
    const std::vector<std::string> PYTORCH_MLU_ALLOC_CONF = {
        "PYTORCH_MLU_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"};
    auto inst = new MLUAllocatorConfig();
    auto str_env = torch_mlu::getCvarString(PYTORCH_MLU_ALLOC_CONF, "");
    const char* env = str_env.c_str();
    inst->parseArgs(env);
    return inst;
  })();
  return *s_instance;
}

// for more info:
// https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
void MLUAllocatorConfig::parseArgs(const char* env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
  m_garbage_collection_threshold = 0;

  if (env == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(m_last_allocator_settings_mutex);
    m_last_allocator_settings = env;
  }

  std::vector<std::string> config;
  lexArgs(env, config);

  for (size_t i = 0; i < config.size(); i++) {
    std::string_view config_item_view(config[i]);
    if (config_item_view == "max_split_size_mb") {
      i = parseMaxSplitSize(config, i);
    } else if (config_item_view == "garbage_collection_threshold") {
      i = parseGarbageCollectionThreshold(config, i);
    } else if (config_item_view == "roundup_power2_divisions") {
      i = parseRoundUpPower2Divisions(config, i);
    } else if (config_item_view == "expandable_segments") {
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for expandable_segments");
      config_item_view = config[i];
      m_expandable_segments = (config_item_view == "True");
    } else if (config_item_view == "release_lock_on_cnrtmalloc") {
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for release_lock_on_cnrtmalloc");
      config_item_view = config[i];
      m_release_lock_on_cnrtmalloc = (config_item_view == "True");
    } else if (config_item_view == "use_linear_memory") {
      consumeToken(config, ++i, ':');
      ++i;
      TORCH_CHECK(
          i < config.size() &&
              (std::string_view(config[i]) == "True" ||
               std::string_view(config[i]) == "False"),
          "Expected a single True/False argument for use_linear_memory");
      config_item_view = config[i];
      m_use_linear_memory = (config_item_view == "True");
    } else {
      TORCH_CHECK(false, "Unrecognized CachingAllocator option: ", config[i]);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }
}

// General caching allocator utilities
void setAllocatorSettings(const std::string& env) {
  MLUCachingAllocator::MLUAllocatorConfig::instance().parseArgs(env.c_str());
}

} // namespace torch_mlu::MLUCachingAllocator
