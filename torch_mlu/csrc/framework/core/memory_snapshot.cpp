#include <ATen/Context.h>
#include "framework/core/caching_allocator.h"
#include "framework/core/memory_snapshot.h"
#include <torch/csrc/profiler/combined_traceback.h>

namespace torch_mlu {

std::shared_ptr<c10::GatheredContext> gather() {
  return torch::CapturedTraceback::gather(true, true, false);
}

std::shared_ptr<c10::GatheredContext> gather_with_cpp() {
  return torch::CapturedTraceback::gather(true, true, true);
}

void _record_memory_history(
    bool enabled,
    bool record_context,
    int64_t trace_alloc_max_entries,
    bool trace_alloc_record_context,
    bool record_cpp_context) {
  torch_mlu::MLUCachingAllocator::CreateContextFn recorder = gather;
  if (enabled && record_cpp_context) {
    recorder = gather_with_cpp;
    // warm up C++ stack unwinding
    torch::unwind::unwind();
  }
  auto when = torch_mlu::MLUCachingAllocator::RecordContext::NEVER;
  if (trace_alloc_record_context) {
    when = torch_mlu::MLUCachingAllocator::RecordContext::ALLOC;
  } else if (record_context) {
    when = torch_mlu::MLUCachingAllocator::RecordContext::STATE;
  }
  at::globalContext().lazyInitMLU();
  torch_mlu::MLUCachingAllocator::recordHistory(
      enabled, recorder, trace_alloc_max_entries, when);
}

static void checkOptionIn(
    const std::string& option,
    std::initializer_list<std::string> valid,
    const char* error) {
  TORCH_CHECK(
      valid.end() != std::find(valid.begin(), valid.end(), option), error);
}

void _record_memory_history(
    c10::optional<std::string> enabled,
    c10::optional<std::string> context,
    std::string stacks,
    size_t max_entries) {
  if (enabled) {
    checkOptionIn(
        *enabled,
        {"state", "all"},
        "expected state to be 'state', 'all', or None");
  }
  if (context) {
    checkOptionIn(
        *context,
        {"state", "alloc", "all"},
        "expected context to be 'state', 'alloc', 'all', or None");
  }
  checkOptionIn(
      stacks, {"python", "all"}, "expected stacks to be 'python', or 'all'");

  torch_mlu::MLUCachingAllocator::CreateContextFn recorder = gather;
  if (enabled && stacks == "all") {
    recorder = gather_with_cpp;
    // warm up C++ stack unwinding
    torch::unwind::unwind();
  }
  max_entries = (enabled && *enabled == "all") ? max_entries : 1;
  auto when = torch_mlu::MLUCachingAllocator::RecordContext::NEVER;
  if (context) {
    if (context == "all") {
      when = torch_mlu::MLUCachingAllocator::RecordContext::ALL;
    } else if (context == "alloc") {
      when = torch_mlu::MLUCachingAllocator::RecordContext::ALLOC;
    } else if (context == "state") {
      when = torch_mlu::MLUCachingAllocator::RecordContext::STATE;
    }
  }
  at::globalContext().lazyInitMLU();
  torch_mlu::MLUCachingAllocator::recordHistory(
      enabled.has_value(), recorder, max_entries, when);
}

} // namespace torch_mlu
