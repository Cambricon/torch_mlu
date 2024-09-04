#include "profile_mlu.h"

#include <atomic>
#include <memory>
#include <list>
#include <unordered_map>
#include <algorithm>

#include "libkineto.h"
#include "ActivityProfilerProxy.h"
#include "Logger.h"

using namespace KINETO_NAMESPACE;
namespace torch_mlu { namespace profiler {

void enableMluProfiler() {
  if (!libkineto::api().isProfilerRegistered()) {
    ConfigLoader& config_loader = libkineto::api().configLoader();
    libkineto::api().registerProfiler(
        std::make_unique<ActivityProfilerProxy>(/* cpuOnly = */false, config_loader));
    libkineto::api().suppressLogMessages();
  }
}
}  // namespace profiler
}  // namespace torch_mlu
