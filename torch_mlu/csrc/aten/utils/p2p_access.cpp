#include <torch/csrc/utils/mlu_lazy_init.h>
#include <c10/util/irange.h>

#include "aten/utils/p2p_access.h"
#include "framework/core/caching_allocator.h"

#include <vector>

namespace torch_mlu {

namespace {

static std::vector<int8_t> p2pAccessEnabled;
static int64_t num_dev = -1;

} // namespace

void init_p2p_access_cache(int64_t num_devices) {
  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  // Currently the max number of gpus in P2P group is 8, so if there are more
  // we enable P2P in groups of 8
  p2pAccessEnabled.clear();
  p2pAccessEnabled.resize(num_devices * num_devices, -1);
  num_dev = num_devices;

  for (const auto i : c10::irange(num_devices)) {
    p2pAccessEnabled[i * num_devices + i] = 1;
  }
}

bool get_p2p_access(int dev, int dev_to_access) {
  at::globalContext().lazyInitMLU();

  TORCH_CHECK(dev >= 0 || dev < num_dev, dev, " is not a device");
  TORCH_CHECK(
      dev_to_access >= 0 || dev_to_access < num_dev,
      dev_to_access,
      " is not a device");
  TORCH_INTERNAL_ASSERT(num_dev >= 0, "p2p access cache not initialized");

  auto& cache = p2pAccessEnabled[dev * num_dev + dev_to_access];

  if (cache != -1) {
    return cache;
  }

  unsigned int result = 0;
  CNRT_CHECK(cnrtGetPeerAccessibility(&result, dev, dev_to_access));
  cache = result ? 1 : 0;
  if (cache) {
    MLUCachingAllocator::enablePeerAccess(dev, dev_to_access);
  }

  return cache;
}

} // namespace torch_mlu