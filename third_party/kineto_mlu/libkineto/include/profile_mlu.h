#pragma once

#include <torch/csrc/Export.h>
#include <mutex>
#include <vector>
#include <string>
#include <cstdint>

namespace torch_mlu {
namespace profiler {

TORCH_API void enableMluProfiler();

}  // namespace profiler
}  // namespace torch_mlu
