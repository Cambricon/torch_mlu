#include "framework/core/caching_event.h"

namespace torch_mlu {

CachingMLUEvent CachingMLUEvent::instance;

// Get Singleton instance
CachingMLUEvent& CachingMLUEvent::get_instance() {
  return instance;
}

std::shared_ptr<MLUEvent> CachingMLUEvent::alloc_event(
    c10::DeviceIndex device_index) {
  std::shared_ptr<MLUEvent> sptr(nullptr);
  int device_id = static_cast<int>(device_index);
  TORCH_CHECK(
      0 <= device_id && device_id < MLU_DEVICE_NUM_MAX,
      "Device id is invalid.");
  if (event_pool[device_id].empty()) {
    // explicit disable timing flag.
    sptr = std::make_shared<MLUEvent>(CNRT_NOTIFIER_DISABLE_TIMING_ALL);
  } else {
    std::lock_guard<std::mutex> lock(event_mutex[device_id]);
    sptr = event_pool[device_id].front();
    event_pool[device_id].pop_front();
  }
  return sptr;
}

void CachingMLUEvent::give_back_event(std::shared_ptr<MLUEvent> sptr) {
  int device_id = static_cast<int>(sptr->device_index());
  std::lock_guard<std::mutex> lock(event_mutex[device_id]);
  event_pool[device_id].emplace_back(sptr);
}

void CachingMLUEvent::clean_event() {
  for (int i = 0; i < MLU_DEVICE_NUM_MAX; ++i) {
    for (auto& sptr : event_pool[i]) {
      sptr.reset();
    }
    event_pool[i].clear();
  }
}

} // namespace torch_mlu
