#include <torch/csrc/profiler/standalone/privateuse1_observer.h>
#include <torch/csrc/profiler/orchestration/observer.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>
#include <ATen/record_function.h>
#include <fmt/format.h>

namespace torch_mlu {
namespace profiler {
namespace impl {

using namespace torch::profiler::impl;

struct CNPXThreadLocalState : ProfilerStateBase {
  explicit CNPXThreadLocalState(const ProfilerConfig& config)
      : ProfilerStateBase(config) {
    // Only `report_input_shapes` makes sense in this context.
    TORCH_CHECK(!config.profile_memory);
    TORCH_CHECK(!config.with_stack);
    TORCH_CHECK(!config.with_flops);
    TORCH_CHECK(!config.with_modules);
  }
  ~CNPXThreadLocalState() override = default;

  ActiveProfilerType profilerType() override {
    return ActiveProfilerType::PRIVATEUSE1;
  }

  void reportMemoryUsage(void*, int64_t, size_t, size_t, c10::Device) override {
  }

  static CNPXThreadLocalState* getTLS() {
    auto tls = ProfilerStateBase::get(/*global=*/false);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        tls == nullptr ||
        tls->profilerType() == ActiveProfilerType::PRIVATEUSE1);
    return static_cast<CNPXThreadLocalState*>(tls);
  }
  std::pair<at::RecordFunctionHandle, int> getOpIdFromInput(
      const at::Tensor& tensor);

  void setProducerTensorMap(
      at::TensorImpl* tensor,
      at::RecordFunctionHandle op_id,
      int output_nr) {
    producer_tensor_map_[(void*)tensor] =
        std::pair<at::RecordFunctionHandle, int>{op_id, output_nr};
  }

 protected:
  // Maps the address of an output Tensor to a unique op id and output
  // index of the tensor.
  // at::TensorImpl* is the actual type of the key, but using void*
  // to indicate the pointer is just being used as a key
  std::unordered_map<void*, std::pair<at::RecordFunctionHandle, int>>
      producer_tensor_map_;
};

std::pair<at::RecordFunctionHandle, int> CNPXThreadLocalState::getOpIdFromInput(
    const at::Tensor& tensor) {
  std::pair<at::RecordFunctionHandle, int> producer_op_pair(0, -1);
  if (tensor.defined()) {
    at::TensorImpl* ten_addr = tensor.unsafeGetTensorImpl();
    // See if Address is in the map already
    if (producer_tensor_map_.count((void*)ten_addr) > 0) {
      producer_op_pair = producer_tensor_map_[(void*)ten_addr];
    }
  }
  return producer_op_pair;
}

namespace {
std::list<std::pair<at::RecordFunctionHandle, int>> flattenOpIdList(
    const c10::List<c10::IValue>& list) {
  std::list<std::pair<at::RecordFunctionHandle, int>> input_op_id_list;
  auto state_ptr = CNPXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& input : list) {
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      auto producer_op_pair = state_ptr->getOpIdFromInput(tensor);
      input_op_id_list.push_back(producer_op_pair);
    }
  }
  return input_op_id_list;
}

std::list<std::pair<at::RecordFunctionHandle, int>> getInputTensorOpIds(
    const at::RecordFunction& fn) {
  std::pair<at::RecordFunctionHandle, int> undefined_op_pair(0, -1);
  std::list<std::pair<at::RecordFunctionHandle, int>> input_producer_ops_;
  auto state_ptr = CNPXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& input_item : fn.inputs()) {
    if (input_item.isTensor()) {
      const at::Tensor& tensor = input_item.toTensor();
      auto producer_pair = state_ptr->getOpIdFromInput(tensor);
      input_producer_ops_.push_back(producer_pair);
    } else {
      if (input_item.isList()) {
        std::list<std::pair<at::RecordFunctionHandle, int>> tmp_op_ids =
            flattenOpIdList(input_item.toList());
        // Extend the current sizes array by the array returned from input sizes
        if (!tmp_op_ids.empty()) {
          input_producer_ops_.splice(input_producer_ops_.end(), tmp_op_ids);
        } else {
          input_producer_ops_.emplace_back(undefined_op_pair);
        }
      } else {
        input_producer_ops_.emplace_back(undefined_op_pair);
      }
    }
  }
  return input_producer_ops_;
}

void updateOutputTensorTracker(const at::RecordFunction& fn) {
  int output_nr = 0;
  auto state_ptr = CNPXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");
  for (const c10::IValue& s_tensor : fn.outputs()) {
    if (s_tensor.isTensor()) {
      const at::Tensor& tensor = s_tensor.toTensor();
      if (tensor.defined()) {
        auto ten_addr = tensor.unsafeGetTensorImpl();
        state_ptr->setProducerTensorMap(ten_addr, fn.handle(), output_nr);
      }
    }
    output_nr++;
  }
}
} // namespace

std::string getCnpxStr(
    const char* name,
    int64_t sequence_nr,
    const std::vector<std::vector<int64_t>>& shapes,
    at::RecordFunctionHandle op_id,
    const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids) {
  if (sequence_nr >= -1 || !shapes.empty()) {
    std::string str;
    if (sequence_nr >= 0) {
      str = fmt::format("{}, seq = {}", name, sequence_nr);
    } else if (sequence_nr == -1) {
      str = name;
    }
    if (op_id > 0) {
      str = fmt::format("{}, op_id = {}", str, op_id);
    }
    if (!shapes.empty()) {
      str = fmt::format("{}, sizes = {}", str, shapesToStr(shapes));
    }
    // Include the op ids of the input edges so
    // you can build the network graph
    if (!input_op_ids.empty()) {
      str = fmt::format(
          "{}, input_op_ids = {}", str, inputOpIdsToStr(input_op_ids));
    }
    return str;
  } else {
    return name;
  }
}

template <bool report_input_shapes>
std::unique_ptr<at::ObserverContext> enterCNPX(const at::RecordFunction& fn) {
  if (CNPXThreadLocalState::getTLS() != nullptr) {
    auto input_op_ids = getInputTensorOpIds(fn);
    torch::profiler::impl::privateuse1Stubs()->rangePush(
        getCnpxStr(
            fn.name(),
            fn.seqNr(),
            report_input_shapes ? torch::profiler::impl::inputSizes(fn, true)
                                : std::vector<std::vector<int64_t>>(),
            fn.handle(),
            report_input_shapes
                ? input_op_ids
                : std::list<std::pair<at::RecordFunctionHandle, int>>())
            .c_str());
  }
  return nullptr;
}

void pushCNPXCallbacks(
    const ProfilerConfig& config,
    const std::unordered_set<at::RecordScope>& scopes) {
  TORCH_CHECK(
      torch::profiler::impl::privateuse1Stubs()->enabled(),
      "Can't use CNPX profiler - CATCH was not compiled");

  c10::ThreadLocalDebugInfo::_push(
      c10::DebugInfoKind::PROFILER_STATE,
      std::make_shared<torch_mlu::profiler::impl::CNPXThreadLocalState>(
          config));

  auto state_ptr = torch_mlu::profiler::impl::CNPXThreadLocalState::getTLS();
  TORCH_INTERNAL_ASSERT(state_ptr, "Expected profiler state set");

  auto handle = at::addThreadLocalCallback(
      at::RecordFunctionCallback(
          state_ptr->config().report_input_shapes
              ? &torch_mlu::profiler::impl::enterCNPX<
                    /*report_input_shapes=*/true>
              : &torch_mlu::profiler::impl::enterCNPX<
                    /*report_input_shapes=*/false>,
          [](const at::RecordFunction& fn, at::ObserverContext* ctx) {
            torch::profiler::impl::privateuse1Stubs()->rangePop();
            torch_mlu::profiler::impl::updateOutputTensorTracker(fn);
          })
          .needsInputs(config.report_input_shapes)
          .needsOutputs(config.report_input_shapes)
          .needsIds(true)
          .scopes(scopes));
  state_ptr->setCallbackHandle(handle);
}

} // namespace impl
} // namespace profiler
} // namespace torch_mlu

namespace torch {
namespace profiler {
namespace impl {
REGISTER_PRIVATEUSE1_OBSERVER(
    pushPRIVATEUSE1CallbacksStub,
    &torch_mlu::profiler::impl::pushCNPXCallbacks);
}
} // namespace profiler
} // namespace torch