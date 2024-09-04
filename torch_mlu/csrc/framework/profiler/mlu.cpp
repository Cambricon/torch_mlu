#include <cnpx.h>
#include <torch/csrc/autograd/profiler.h>

using namespace torch::profiler::impl;

namespace torch_mlu {
namespace profiler {
namespace impl {
namespace {

struct MLUMethods : public ProfilerStubs {
  void record(
      c10::DeviceIndex* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {
    AT_ERROR("MLUMethods::record() is not implemented.");
  }

  float elapsed(
      const ProfilerVoidEventStub* event,
      const ProfilerVoidEventStub* event2) const override {
    AT_ERROR("MLUMethods::elapsed() is not implemented.");
  }

  void mark(const char* name) const override {
    cnpxMark(name);
  }

  void rangePush(const char* name) const override {
    cnpxRangePush(name);
  }

  void rangePop() const override {
    cnpxRangePop();
  }

  void onEachDevice(std::function<void(int)> op) const override {
    AT_ERROR("MLUMethods::onEachDevice() is not implemented.");
  }

  void synchronize() const override {
    AT_ERROR("MLUMethods::synchronize() is not implemented.");
  }

  bool enabled() const override {
    return true;
  }
};

struct RegisterMLUMethods {
  RegisterMLUMethods() {
    static MLUMethods methods;
    registerMLUMethods(&methods);
  }
};
RegisterMLUMethods reg;

} // namespace
} // namespace impl
} // namespace profiler
} // namespace torch_mlu
