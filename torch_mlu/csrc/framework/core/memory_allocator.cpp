/*
All modification made by Cambricon Corporation: © 2022 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2022, the respective contributors
All rights reserved.
For the list of contributors go to
https://github.com/pytorch/pytorch/graphs/contributors Redistribution and use in
source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "framework/core/MLUEvent.h"
#include "framework/hooks/MLUHooks.h"
#include "framework/core/caching_event.h"
#include "framework/core/memory_allocator.h"

namespace torch_mlu {

namespace {

using Block = at::HostBlock<MLUStream>;

struct MLUCachingHostAllocatorImpl
    : public at::
          CachingHostAllocatorImpl<MLUStream, std::shared_ptr<MLUEvent>> {
 private:
  void allocate_host_memory(size_t size, void** ptr) override {
    // Pinned memory pointers allocated by any device can be directly used by
    // any other device, regardless of the current device at the time of
    // allocation, unified addressing. So we grab any existing primary context,
    // if available. See pytorch/pytorch#21081.
    at::OptionalDeviceGuard device_guard;
    auto shared_context_device_index =
        torch_mlu::getDeviceIndexWithSharedContext();
    if (shared_context_device_index.has_value()) {
      device_guard.reset_device(
          at::Device(at::kPrivateUse1, *shared_context_device_index));
    }
    // cnrtHostMalloc is different with cudaHostAlloc when size is 0.
    // cnrtHostMalloc will return nullptr when size is 0.
    TORCH_CNRT_CHECK(cnrtHostMalloc(ptr, size));
  }

  void free_block(Block* block) override {
    TORCH_CNRT_CHECK(cnrtFreeHost(block->ptr_));
  }

  void record_stream(
      std::optional<std::vector<std::shared_ptr<MLUEvent>>>& events,
      MLUStream stream) override {
    auto event_sptr = MLUEventPool_Manager.alloc_event(stream.device_index());
    event_sptr->place(stream);
    events->emplace_back(std::move(event_sptr));
  }

  bool query_event(std::shared_ptr<MLUEvent>& event) override {
    const bool ret = event->query();
    if (ret) {
      // Give back event to event pool and pop event
      // output of event-ptr deque.
      MLUEventPool_Manager.give_back_event(event);
    }
    return ret;
  }
};

void raw_local_deleter(void* ptr);

// Malloc a pin memory through cnrt interface.
struct MLUCachingHostAllocator final
    : public at::CachingHostAllocatorInterface<MLUCachingHostAllocatorImpl> {
  at::DataPtr allocate(size_t size) override {
    auto ptr_and_ctx = impl_->allocate(size);
    return {
        ptr_and_ctx.first,
        ptr_and_ctx.second,
        &raw_local_deleter,
        at::DeviceType::CPU};
  }
};

static MLUCachingHostAllocator mlu_caching_host_allocator;

static inline MLUCachingHostAllocator& getMLUCachingHostAllocator() {
  return mlu_caching_host_allocator;
}

void raw_local_deleter(void* ptr) {
  getMLUCachingHostAllocator().free(ptr);
}

} // anonymous namespace

bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    torch_mlu::MLUStream stream) {
  return getMLUCachingHostAllocator().record_event(ptr, ctx, stream);
}

void CachingHostAllocator_emptyCache() {
  getMLUCachingHostAllocator().empty_cache();
}

at::Allocator* getCachingHostAllocator() {
  return &getMLUCachingHostAllocator();
}

} // namespace torch_mlu
