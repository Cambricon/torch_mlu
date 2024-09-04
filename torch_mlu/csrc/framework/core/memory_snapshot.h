#pragma once

#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>
#include <string>

namespace torch_mlu {

// C++-only versions of these, for python use
// those defined in cuda/Module.cpp which also record python state.
TORCH_MLU_API void _record_memory_history(
    bool enabled,
    bool record_context = true,
    int64_t trace_alloc_max_entries = 1,
    bool trace_alloc_record_context = false,
    bool record_cpp_context = false);

TORCH_MLU_API void _record_memory_history(
    std::optional<std::string> enabled = "all",
    std::optional<std::string> context = "all",
    std::string stacks = "all",
    size_t max_entries = UINT64_MAX);

} // namespace torch_mlu
