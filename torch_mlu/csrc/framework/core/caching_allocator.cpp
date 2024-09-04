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

#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>
#include <c10/util/static_tracepoint.h>
#if USE_PROFILE
#include <c10/util/ThreadLocalDebugInfo.h>
#endif
#include <cxxabi.h>
#include <execinfo.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>
#include <regex>

#include "framework/core/caching_allocator.h"
#include "framework/core/caching_allocator_config.h"
#include "framework/core/device.h"
#include "framework/core/mlu_guard.h"
#include "framework/core/caching_event.h"
#include "framework/graphs/MLUGraphUtils.h"

TORCH_SDT_DEFINE_SEMAPHORE(malloc)
TORCH_SDT_DEFINE_SEMAPHORE(free)

namespace torch_mlu::MLUCachingAllocator {

C10_DEFINE_REGISTRY(FreeMluMemoryCallbacksRegistry, FreeMemoryCallback);

// Included here as this is externally used in MLUAllocatorConfig
const size_t large_buffer_size_mlu =
    67108864; // "large" allocations may be in 64 MiB chunks

namespace Native {

// the constant parameters for chunk
constexpr size_t minimum_round_size =
    512; // all chunks are rounded at least 512 bytes
constexpr size_t small_allocation_size =
    1048576; // maximum for "small" allocation is 1 Mib
constexpr size_t small_buffer_size =
    2097152; // "small" allocations are in 2 Mibs chunks
constexpr size_t large_allocation_size =
    10485760; // allocation sizes from 1 Mib to 10 Mibs use larger chunks
constexpr size_t large_buffer_size =
    20971520; // "large" allocations may be in 20 Mibs chunks
constexpr size_t maximum_round_size =
    2097152; // all chunks are rounded at most 2 Mibs

// the chunk's constant parameters for MLU
constexpr size_t small_allocation_size_mlu =
    16777216; // maximum for "small" allocation is 16 MiB
constexpr size_t small_buffer_size_mlu =
    33554432; // "small" allocations are in 32 MiB chunks
constexpr size_t large_allocation_size_mlu =
    33554432; // allocation sizes from 16 MiB to 32 MiB use larger chunks

namespace {

using stream_set = ska::flat_hash_set<torch_mlu::MLUStream>;

using StatTypes = std::array<bool, static_cast<size_t>(StatType::NUM_TYPES)>;

void update_stat(Stat& stat, int64_t amount) {
  stat.current += amount;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      stat.current >= 0,
      "Negative tracked stat in MLU allocator (likely logic error).");

  stat.peak = std::max(stat.current, stat.peak);
  if (amount > 0) {
    stat.allocated += amount;
  }
  if (amount < 0) {
    stat.freed += -amount;
  }
}

void reset_accumulated_stat(Stat& stat) {
  stat.allocated = 0;
  stat.freed = 0;
}

void reset_peak_stat(Stat& stat) {
  stat.peak = stat.current;
}

template <typename Func>
void for_each_selected_stat_type(const StatTypes& stat_types, Func f) {
  for (const auto stat_type : c10::irange(stat_types.size())) {
    if (stat_types[stat_type]) {
      f(stat_type);
    }
  }
}

void update_stat_array(
    StatArray& stat_array,
    int64_t amount,
    const StatTypes& stat_types) {
  for_each_selected_stat_type(
      stat_types, [&stat_array, amount](size_t stat_type) {
        update_stat(stat_array[stat_type], amount);
      });
}

// ChunkPool is a sorted list of Chunk, using pointer for comparing
struct Chunk;
struct PrivatePool;
typedef bool (*Comparison)(const Chunk*, const Chunk*);
static bool ChunkComparatorSize(const Chunk* a, const Chunk* b);
static bool ChunkComparatorAddress(const Chunk* a, const Chunk* b);

struct ChunkPool {
  ChunkPool(bool small, PrivatePool* private_pool = nullptr)
      : chunks(ChunkComparatorSize),
        unmapped(ChunkComparatorAddress),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  // Do not insert a Chunk to chunks directly; use insert_into_chunks(),
  // instead.
  std::set<Chunk*, Comparison> chunks;
  std::set<Chunk*, Comparison> unmapped;
  const bool is_small;
  PrivatePool* owner_PrivatePool;
  int64_t get_free_chunks_call_count{0};

  // Add a Chunk into Chunks set with updating gc counter.
  std::pair<std::set<Chunk*, Comparison>::iterator, bool> insert_into_chunks(
      Chunk* chunk);
};

struct ExpandableSegment;

struct Chunk {
  int device; // mlu device id
  cnrtQueue_t stream; // allocation stream
  stream_set stream_uses; // streams on which the chunk was used
  size_t size; // chunk size in bytes
  size_t requested_size; // memory originally requested
  ChunkPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in_use flag
  bool mapped{true}; // is the virtual address range this Chunk references
                     // backed by physical pages. Always true when
                     // expandable_segment_ is null. When false
                     // This Chunk will be aligned to the segment size
                     // of its expandable_segment_.
  Chunk* prev{nullptr}; // prev chunk if split from a larger allocation
  Chunk* next{nullptr}; // next chunk if split from a larger allocation
  int event_count{0}; // number of outstanding MLU events.
  int gc_count_base{0}; // get_free_chunks_call_count when Chunk is inserted
  std::shared_ptr<c10::GatheredContext> context_when_allocated;
  // only set for the first chunk in the segment (when prev == null)
  // this records the frame information when cnrtMalloc was called
  // whereas context_when_allocated records the last time we handed this
  // memory out from our cache.
  std::shared_ptr<c10::GatheredContext> context_when_segment_allocated;

  ExpandableSegment* expandable_segment_{nullptr};

  Chunk(int device, cnrtQueue_t stream, size_t size, ChunkPool* pool, void* ptr)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr),
        gc_count_base(0) {}

  // constructor for search key
  Chunk(int device, cnrtQueue_t stream, size_t size)
      : device(device),
        stream(stream),
        stream_uses(),
        size(size),
        requested_size(0) {}

  size_t gc_count() {
    TORCH_INTERNAL_ASSERT(pool);
    return static_cast<int>(pool->get_free_chunks_call_count - gc_count_base);
  }

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
  void splice(Chunk* before, Chunk* after) {
    if (before) {
      TORCH_INTERNAL_ASSERT(before->next == after);
      before->next = this;
    }
    prev = before;
    if (after) {
      TORCH_INTERNAL_ASSERT(after->prev == before);
      after->prev = this;
    }
    next = after;
  }
};

std::pair<std::set<Chunk*, Comparison>::iterator, bool> ChunkPool::
    insert_into_chunks(Chunk* chunk) {
  chunk->gc_count_base = get_free_chunks_call_count;
  return chunks.insert(chunk);
}

struct SegmentRange {
  char* ptr;
  size_t size;
  SegmentRange(void* p, size_t s) : ptr(static_cast<char*>(p)), size(s) {}
};

struct ExpandableSegment {
  ExpandableSegment(
      int device,
      cnrtQueue_t stream,
      size_t size,
      const std::vector<int>& peers) {
    TORCH_INTERNAL_ASSERT(false, "expandable segment not supported");
  }
  SegmentRange map(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  SegmentRange unmap(SegmentRange range) {
    return SegmentRange(nullptr, 0);
  }
  char* ptr() const {
    return nullptr;
  }
  size_t size() const {
    return 0;
  }
  void addPeer(int device) {}
};

// ChunkState, ChunkPoolState, and PrivatePoolState contain the information
// needed to reconstruct a private pool to a previous state. See note
// [Checkpointing PrivatePoolState]
struct ChunkState {
  int device = 0;
  cnrtQueue_t stream = nullptr;
  stream_set stream_uses = {};
  size_t size = 0;
  void* ptr = nullptr;
  bool allocated = false;
  int64_t gc_count_base = 0;
  // maintain invariant that event_count == 0 ;
  // history will be left alone in checkpoint
  ChunkState(Chunk* chunk);
};

struct SegmentState {
  std::vector<ChunkState> chunks;
  bool is_small = false;
  SegmentState(Chunk* head);
};

struct PrivatePoolState : AllocatorState {
  // omitting use_count, and cnrtMalloc_count as they remain the same
  MempoolId_t owner_id = {0, 0};

  std::vector<SegmentState> segments;

  PrivatePoolState(
      MempoolId_t pool_id,
      const std::vector<Chunk*>& private_pool_head_chunks);
};

struct RestoreResult {
  std::vector<void*> allocations_freed;
  std::vector<Chunk*> allocations_created;
};

static bool ChunkComparatorSize(const Chunk* a, const Chunk* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}
static bool ChunkComparatorAddress(const Chunk* a, const Chunk* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams {
  AllocParams(
      int device,
      size_t size,
      cnrtQueue_t stream,
      ChunkPool* pool,
      size_t alloc_size,
      MemoryStats& stats)
      : search_key(device, stream, size),
        pool(pool),
        alloc_size(alloc_size),
        chunk(nullptr),
        err(cnrtSuccess) {}

  int device() const {
    return search_key.device;
  }
  cnrtQueue_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Chunk search_key;
  ChunkPool* pool;
  size_t alloc_size;
  Chunk* chunk;
  StatTypes stat_types = {false};
  cnrtRet_t err;
};

// MLU graphs helper
struct PrivatePool {
  PrivatePool()
      : use_count(1),
        cnrtMalloc_count(0),
        large_chunks(/*small=*/false, this),
        small_chunks(/*small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  // Number of live graphs using this pool
  int use_count;
  // Number of unfreed cnrtMallocs made for this pool. When use_count and
  // cnrtMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int cnrtMalloc_count;
  // Instead of maintaining private ChunkPools here, I could stuff all chunks
  // (private or no) into the top-level large_chunks and small_chunks, and
  // distinguish private chunks by adding a "pool id" check above the stream
  // check in ChunkComparator. ChunkComparator is performance- critial though,
  // I'd rather not add more logic to it.
  ChunkPool large_chunks;
  ChunkPool small_chunks;
};

ChunkState::ChunkState(Chunk* chunk)
    : stream(chunk->stream),
      stream_uses(chunk->stream_uses),
      size(chunk->size),
      ptr(chunk->ptr),
      allocated(chunk->allocated),
      gc_count_base(chunk->gc_count_base) {
  TORCH_CHECK(
      chunk->event_count == 0,
      "MLUEvents should have synchronized when checkpointing chunk");
};

SegmentState::SegmentState(Chunk* head) {
  TORCH_INTERNAL_ASSERT(head->prev == nullptr && head->pool != nullptr);
  is_small = head->pool->is_small;

  for (Chunk* curr = head; curr != nullptr; curr = curr->next) {
    chunks.emplace_back(curr);
  }
}

PrivatePoolState::PrivatePoolState(
    MempoolId_t pool_id,
    const std::vector<Chunk*>& private_pool_head_chunks)
    : owner_id(std::move(pool_id)) {
  for (Chunk* head : private_pool_head_chunks) {
    segments.emplace_back(head);
  }
}

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

cnrtRet_t cnrtMallocMaybeCapturing(void** p, size_t size) {
  if (torch_mlu::currentStreamCaptureStatusMayInitCtx() ==
      torch_mlu::CaptureStatus::None) {
    return cnrtMalloc(p, size);
  } else {
    // It's ok to capture cnrtMallocs, as long as we never cnrtFree those
    // addresses before replay.
    // Capturing cnrtMalloc behaves nicely: it gives the graph new VA,
    // but is ignored (won't leakily allocate new memory) in replays.
    torch_mlu::MLUQueueCaptureModeGuard g{cnrtQueueCaptureModeRelaxed};
    return cnrtMalloc(p, size);
  }
}

} // anonymous namespace
} // namespace Native

static std::string reportProcessMemoryInfo(int device) {
  return "";
}

namespace Native {

class DeviceCachingAllocator {
 protected:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // Memory statistics
  MemoryStats stats;

  // cached chunks are larger than 1 MB
  ChunkPool large_chunks;

  // cached chunks are 1 MB or smaller
  ChunkPool small_chunks;

  // allocated or in use by a stream.
  ska::flat_hash_set<Chunk*> active_chunks;

  // captures_underway tracks if we are diverting some
  // allocations to a specific pool.
  // Most of the time it's empty, in which case malloc can avoid calling
  // cnrtQueueGetCaptureInfo in the hot path.
  std::vector<std::pair<MempoolId_t, std::function<bool(cnrtQueue_t)>>>
      captures_underway;
  // See free() for this thing's purpose
  std::vector<Chunk*> needs_events_deferred_until_no_capture;

  // outstanding mlu events
  ska::flat_hash_map<
      torch_mlu::MLUStream,
      std::deque<std::pair<std::shared_ptr<MLUEvent>, Chunk*>>>
      events;

  // record used memory
  size_t total_allocated_memory = 0;

  size_t allowed_memory_maximum = 0;

  // all live expandable segments
  std::vector<ExpandableSegment*> expandable_segments_;
  std::vector<int> devices_with_peer_access_;

  bool set_fraction = false;

  bool record_history = false;

  bool supports_linear_memory = false;

  std::atomic<CreateContextFn> context_recorder_;
  size_t alloc_trace_next = 0;
  RecordContext record_context_ = RecordContext::NEVER;
  size_t alloc_trace_max_entries_ = 1;
  std::vector<TraceEntry>*
      alloc_trace; // pointer because we need to intentionally leak this on
                   // deallocation it can hold references to Python state which
                   // will already be destroyed when we are in exit handlers

  std::vector<AllocatorTraceTracker> trace_trackers_;

  // Members specific to MLU graphs

  // Private pools for MLU graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their ChunkPools are eligible for
  // free_chunks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

 public:
  DeviceCachingAllocator()
      : large_chunks(/*is_small=*/false),
        small_chunks(/*is_small*/ true),
        alloc_trace(new std::vector<TraceEntry>()) {
    stats.max_split_size = MLUAllocatorConfig::max_split_size();
    context_recorder_.store(nullptr);

    int device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&device));
    DeviceProp* prop = torch_mlu::getDeviceProperties(device);
    if (prop->supports_linear_memory) {
      supports_linear_memory = true;
    } else {
      TORCH_WARN_ONCE(
          "Linear memory is not supported on this device. "
          "Falling back to common memory.");
    }
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(when == RecordContext::NEVER || context_recorder);
    record_history = enabled;
    context_recorder_.store(record_history ? context_recorder : nullptr);
    alloc_trace_max_entries_ = std::max(size_t(1), alloc_trace_max_entries);
    record_context_ = enabled ? when : RecordContext::NEVER;
    alloc_trace_next = 0;
    alloc_trace->clear();
  }

  bool isHistoryEnabled() {
    return record_history;
  }

  bool checkPoolLiveAllocations(
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) {
    std::unique_lock<std::recursive_mutex> lock(mutex);

    PrivatePool* pool = nullptr;
    auto pool_it = graph_pools.find(mempool_id);
    TORCH_CHECK(pool_it != graph_pools.end(), "Could not find pool of id");
    pool = pool_it->second.get();

    size_t allocated_pool_chunks = 0;

    for (Chunk* b : active_chunks) {
      if (b->allocated && b->pool->owner_PrivatePool == pool) {
        if (!expected_live_allocations.count(b->ptr)) {
          return false;
        }
        allocated_pool_chunks += 1;
      }
    }

    return allocated_pool_chunks == expected_live_allocations.size();
  }

  void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) {
    std::unique_lock<std::recursive_mutex> lock(mutex);
    trace_trackers_.emplace_back(std::move(tracker));
  }

  // Must be called outside of `mutex` or deadlocks are possible with Python
  std::shared_ptr<c10::GatheredContext> maybeGatherContext(
      RecordContext level) {
    if (record_context_ < level) {
      return nullptr;
    }
    return context_recorder_.load()();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Chunk* malloc(int device, size_t orig_size, cnrtQueue_t stream) {
    // done outside the lock because we don't know what locks the recorder needs
    // to have...
    auto context = maybeGatherContext(RecordContext::STATE);

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway.size() == 0)) {
      // Processes end-of-life events for outstanding allocations used on
      // multiple streams (checks if their MLU-side uses are complete and
      // recycles their memory if so)
      //
      // Q. Why skip process_events if a capture might be underway?
      // A. process_events involves cnrtNotifierQueries, illegal during MLU
      // graph
      //    capture.
      //    Dumb simple solution: defer reclaiming these allocations until after
      //    capture. Cross-stream memory use is uncommon, so the deferral's
      //    effect on memory use during capture should be small.
      process_events(context);
    }

    size_t size = roundUpSize(orig_size);
    auto& pool = get_pool(size, stream);
    size_t alloc_size = getAllocationSize(size);
    AllocParams params(device, size, stream, &pool, alloc_size, stats);
    params.stat_types = get_stat_types_for_pool(pool);

    // First, try to get a chunk from the existing pool.
    bool chunk_found =
        // search pool
        getFreeChunk(params)
        // Trigger callbacks and retry search
        || (trigger_free_memory_callbacks(params) && getFreeChunk(params));

    // Can't reuse existing chunk, try to get a new one.
    if (!chunk_found) {
      // Do garbage collection if the flag is set.
      if (C10_UNLIKELY(
              set_fraction &&
              MLUAllocatorConfig::garbage_collection_threshold() > 0.0)) {
        garbage_collect_cached_chunks(context);
      }
      // Attempt allocate
      // WARNING: alloc_chunk may release the allocator lock when calling
      // cnrtMalloc. So far this function has not modified allocator state, but
      // keep in mind that any observed allocator state may change across calls
      // to alloc_chunk since it may release the lock.
      chunk_found = alloc_chunk(params, false, context)
          // Free enough available cached chunks to satisfy alloc and retry
          // alloc.
          || (free_available_cached_chunks(params, context) &&
              alloc_chunk(params, false, context))
          // Free all non-split cached blocks and retry alloc.
          ||
          (C10_LIKELY(captures_underway.size() == 0) &&
           free_cached_chunks(context) && alloc_chunk(params, true, context));
    }

    if (!chunk_found) {
      // For any error code other than cnrtErrorNoMem,
      // alloc_chunk should have thrown an exception already.
      TORCH_INTERNAL_ASSERT(params.err == cnrtErrorNoMem);

      size_t device_free = 0;
      size_t device_total = 0;
      TORCH_CNRT_CHECK(cnrtMemGetInfo(&device_free, &device_total));
      std::string allowed_info;

      if (set_fraction) {
        allowed_info = format_size(allowed_memory_maximum) + " allowed; ";
      }

      std::string proc_info = reportProcessMemoryInfo(device);

      record_trace(
          TraceEntry::OOM,
          device_free,
          params.size(),
          params.stream(),
          params.device(),
          std::move(context));

      stats.num_ooms += 1;

      c10::reportOutOfMemoryToProfiler(
          size,
          stats.allocated_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          stats.reserved_bytes[static_cast<int64_t>(StatType::AGGREGATE)]
              .current,
          c10::Device(
              c10::DeviceType::PrivateUse1, static_cast<DeviceIndex>(device)));

      auto allocated_bytes =
          stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;
      auto reserved_bytes =
          stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)]
              .current;

      // Make sure we do not have the device lock before calling our
      // observers which might need hold the GIL
      // It is safe to release at this point because will no longer
      // be reading any allocator state.

      lock.unlock();

      // "total capacity": total global memory on MLU
      // "allowed": memory is allowed to use, which set by fraction.
      // "already allocated": memory allocated by the program using the
      //                      caching allocator
      // "free": free memory as reported by the CNRT API
      // "cached": memory held by the allocator but not used by the program
      //
      // The "allocated" amount  does not include memory allocated outside
      // of the caching allocator, such as memory allocated by other programs
      // or memory held by the driver.
      //
      // The sum of "allocated" + "free" + "cached" may be less than the
      // total capacity due to memory held by the driver and usage by other
      // programs.
      //
      // Note that at this point free_cached_chunks has already returned all
      // possible "cached" memory to the driver. The only remaining "cached"
      // memory is split from a larger chunk that is partially in-use.
      TORCH_CHECK_WITH(
          OutOfMemoryError,
          false,
          "MLU out of memory. Tried to allocate ",
          format_size(alloc_size),
          ". MLU ",
          device,
          " has a total capacity of ",
          format_size(device_total),
          " of which ",
          format_size(device_free),
          "is free. ",
          proc_info,
          "Of the allocated memory ",
          format_size(allocated_bytes),
          " is allocated by PyTorch, and ",
          format_size(reserved_bytes - allocated_bytes),
          " is reserved by PyTorch but unallocated.",
          " If reserved but unallocated memory is large try setting",
          " PYTORCH_MLU_ALLOC_CONF=expandable_segments:True to avoid"
          " fragmentation.  See documentation for Memory Management "
          " (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)");
    }

    bool split_remainder = shouldSplit(params.chunk, params.size());
    return alloc_found_chunk(
        std::move(params), orig_size, std::move(context), split_remainder);
  }

  Chunk* alloc_found_chunk(
      AllocParams params,
      size_t orig_size,
      std::shared_ptr<c10::GatheredContext> context,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(
        params.err == cnrtSuccess && params.chunk != nullptr &&
        params.chunk->ptr != nullptr);
    Chunk* chunk = params.chunk;
    Chunk* remain_chunk = nullptr;

    const bool already_split = chunk->is_split();
    // If a chunk needs to be split, we create a new chunk from old, and update
    // stats
    if (split_remainder) {
      remain_chunk = chunk;
      // create a new chunk from old chunk.
      chunk = new Chunk(device, stream, size, pool, chunk->ptr);
      chunk->expandable_segment_ = remain_chunk->expandable_segment_;
      chunk->prev = remain_chunk->prev;
      if (chunk->prev) {
        chunk->prev->next = chunk;
      }
      chunk->next = remain_chunk;

      remain_chunk->prev = chunk;
      remain_chunk->ptr = static_cast<char*>(remain_chunk->ptr) + size;
      remain_chunk->size -= size;
      // carveMasks(chunk, remain_chunk);
      bool inserted = pool->insert_into_chunks(remain_chunk).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

      if (already_split && !chunk->expandable_segment_) {
        // An already-split inactive chunk is being shrunk by size bytes.
        update_stat_array(
            stats.inactive_split_bytes,
            -static_cast<std::int64_t>(chunk->size),
            params.stat_types);
      } else if (!chunk->expandable_segment_) {
        // A new split inactive chunk is being created from a previously unsplit
        // chunk size remaining->size bytes.
        for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
          update_stat(
              stats.inactive_split_bytes[stat_type],
              static_cast<std::int64_t>(remain_chunk->size));
          update_stat(stats.inactive_split[stat_type], 1);
        });
      }
    } else if (already_split && !chunk->expandable_segment_) {
      // An already-split chunk is becoming active
      for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
        update_stat(
            stats.inactive_split_bytes[stat_type],
            -static_cast<std::int64_t>(chunk->size));
        update_stat(stats.inactive_split[stat_type], -1);
      });
    }

    chunk->allocated = true;
    chunk->requested_size = orig_size;
    chunk->context_when_allocated = std::move(context);
    record_trace(
        TraceEntry::ALLOC,
        int64_t(chunk->ptr),
        orig_size,
        chunk->stream,
        chunk->device,
        chunk->context_when_allocated);

    bool inserted = active_chunks.insert(chunk).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    for_each_selected_stat_type(params.stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], 1);
      update_stat(
          stats.allocated_bytes[stat_type],
          static_cast<std::int64_t>(chunk->size));
      update_stat(stats.active[stat_type], 1);
      update_stat(
          stats.active_bytes[stat_type],
          static_cast<std::int64_t>(chunk->size));
      update_stat(
          stats.requested_bytes[stat_type],
          static_cast<std::int64_t>(chunk->requested_size));
    });
    if (chunk->size >= MLUAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, 1);

    c10::reportMemoryUsageToProfiler(
        chunk->ptr,
        chunk->size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::PrivateUse1, device));

    if (supports_linear_memory && MLUAllocatorConfig::use_linear_memory()) {
      int is_linear = 0;
      cnGetMemAttribute(
          reinterpret_cast<void*>(&is_linear),
          CN_MEM_ATTRIBUTE_ISLINEAR,
          reinterpret_cast<CNaddr>(chunk->ptr));
      if (!is_linear) {
        TORCH_WARN_ONCE(
            "The memory allocated is not linear, which may cause performance degradation "
            "and unkonwn error. You can set PYTORCH_MLU_ALLOC_CONF=use_linear_memory:False "
            "to disable using linear memory.");
      }
    }
    return chunk;
  }

  void free(Chunk* chunk) {
    std::shared_ptr<c10::GatheredContext> context =
        maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    chunk->allocated = false;

    // following logic might modifying underlaying Chunk, causing the size
    // changed. We store ahead for reporting
    auto orig_chunk_ptr = chunk->ptr;
    auto orig_chunk_size = chunk->size;

    StatTypes stat_types = get_stat_types_for_pool(*chunk->pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.allocation[stat_type], -1);
      update_stat(
          stats.allocated_bytes[stat_type],
          -static_cast<std::int64_t>(chunk->size));
    });

    record_trace(
        TraceEntry::FREE_REQUESTED,
        int64_t(chunk->ptr),
        chunk->requested_size,
        chunk->stream,
        chunk->device,
        context ? context : chunk->context_when_allocated);

    if (chunk->size >= MLUAllocatorConfig::max_split_size())
      update_stat(stats.oversize_allocations, -1);

    if (!chunk->stream_uses.empty()) {
      if (C10_UNLIKELY(captures_underway.size())) {
        // It's forbidden to cnrtNotifierQuery an event recorded during MLU
        // graph capture. We conservatively defer recording end-of-life events
        // until the next call to process_events() (which won't happen until no
        // captures are underway)
        needs_events_deferred_until_no_capture.push_back(chunk);
      } else {
        insert_events(chunk);
      }
    } else {
      freeChunk(chunk, context);
    }

    c10::reportMemoryUsageToProfiler(
        orig_chunk_ptr,
        -orig_chunk_size,
        stats.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        stats.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
        c10::Device(c10::DeviceType::PrivateUse1, chunk->device));
  }

  void* getBaseAllocation(Chunk* chunk, size_t* outSize) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    TORCH_CHECK(
        !chunk->expandable_segment_,
        "Tensors allocated with expandable_segments:True cannot be "
        "shared between processes. Consider using "
        "expandable_segments:False in data loading workers via "
        "torch.mlu.memory._set_allocator_settings('expandable_"
        "segments:False')");
    while (chunk->prev) {
      chunk = chunk->prev;
    }
    void* basePtr = chunk->ptr;
    if (outSize) {
      size_t size = 0;
      while (chunk) {
        size += chunk->size;
        chunk = chunk->next;
      }
      *outSize = size;
    }
    return basePtr;
  }

  void recordStream(Chunk* chunk, MLUStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == chunk->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    chunk->stream_uses.insert(stream);
  }

  /** set memory fraction to limit maximum allocated memory **/
  void setMemoryFraction(double fraction) {
    size_t device_free = 0;
    size_t device_total = 0;
    TORCH_CNRT_CHECK(cnrtMemGetInfo(&device_free, &device_total));
    allowed_memory_maximum = static_cast<size_t>(fraction * device_total);
    set_fraction = true;
  }

  /** returns cached chunks to the system allocator **/
  void emptyCache() {
    auto context = maybeGatherContext(RecordContext::ALL);
    std::lock_guard<std::recursive_mutex> lock(mutex);
    free_cached_chunks(context);
  }

  /** Retrieves size of largest unused chunk held by the memory cache **/
  void cacheInfo(size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (*largest ==
        0) { // make an initial guess if a zero *largest is passed in
      size_t tmp_bytes = 0;
      TORCH_CNRT_CHECK(cnrtMemGetInfo(
          largest,
          &tmp_bytes)); // Use free memory as an optimistic initial guess of
                        // *largest
    }
    cache_info_aux(large_chunks, largest);
    cache_info_aux(small_chunks, largest);
    for (const auto& gp : graph_pools) {
      cache_info_aux(gp.second->large_chunks, largest);
      cache_info_aux(gp.second->small_chunks, largest);
    }
  }

  /** Returns a copy of the memory allocator stats for the device **/
  MemoryStats getStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    return stats;
  }

  /** Resets the historical accumulation stats for the device **/
  void resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_accumulated_stat(stats.allocation[statType]);
      reset_accumulated_stat(stats.segment[statType]);
      reset_accumulated_stat(stats.active[statType]);
      reset_accumulated_stat(stats.inactive_split[statType]);
      reset_accumulated_stat(stats.allocated_bytes[statType]);
      reset_accumulated_stat(stats.reserved_bytes[statType]);
      reset_accumulated_stat(stats.active_bytes[statType]);
      reset_accumulated_stat(stats.inactive_split_bytes[statType]);
      reset_accumulated_stat(stats.requested_bytes[statType]);
    }

    stats.num_alloc_retries = 0;
    stats.num_ooms = 0;
    reset_accumulated_stat(stats.oversize_allocations);
    reset_accumulated_stat(stats.oversize_segments);
  }

  /** Resets the historical peak stats for the device **/
  void resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    for (const auto statType :
         c10::irange(static_cast<size_t>(StatType::NUM_TYPES))) {
      reset_peak_stat(stats.allocation[statType]);
      reset_peak_stat(stats.segment[statType]);
      reset_peak_stat(stats.active[statType]);
      reset_peak_stat(stats.inactive_split[statType]);
      reset_peak_stat(stats.allocated_bytes[statType]);
      reset_peak_stat(stats.reserved_bytes[statType]);
      reset_peak_stat(stats.active_bytes[statType]);
      reset_peak_stat(stats.inactive_split_bytes[statType]);
      reset_peak_stat(stats.requested_bytes[statType]);
    }
    reset_peak_stat(stats.oversize_allocations);
    reset_peak_stat(stats.oversize_segments);
  }

  /* Checkpoint the state of a private pool necessary to return it to its
    current state */
  std::unique_ptr<PrivatePoolState> getCheckpointState(MempoolId_t id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    auto pool = graph_pools.find(id);
    if (pool != graph_pools.end()) {
      auto private_pool_head_chunks =
          get_private_pool_head_chunks(pool->second.get());
      return std::make_unique<PrivatePoolState>(id, private_pool_head_chunks);
    } else if (graph_pools_freeable.count(id)) {
      TORCH_CHECK(false, "Not expected to checkpoint freeable graph");
    } else {
      TORCH_CHECK(false, "Could not find pool of id");
    }
  }

  void freeChunksAllocatedToPool(PrivatePool* private_pool, RestoreResult& rr) {
    std::unordered_map<void*, Chunk*> orig_ptrs_to_chunks;

    auto pool_chunks = get_private_pool_head_chunks(private_pool);

    std::vector<Chunk*> head_chunks;
    for (Chunk* chunk : pool_chunks) {
      if (chunk->prev == nullptr) {
        head_chunks.push_back(chunk);
      }
    }

    for (Chunk* chunk : head_chunks) {
      Chunk* curr = chunk;

      while (curr) {
        // When we free a chunk, its pointer should never change
        // only its adjacent chunks, so free, then look at pointer
        if (curr->allocated) {
          TORCH_CHECK(
              curr->event_count == 0,
              "MLUEvents should have synchronized when setting checkpointed chunk");
          rr.allocations_freed.push_back(curr->ptr);
          free(curr);
          TORCH_CHECK(!curr->allocated)
        }
        curr = curr->next;
      }
    }

    for (Chunk* b : get_private_pool_head_chunks(private_pool)) {
      Chunk* curr = b;
      while (curr) {
        TORCH_CHECK(!curr->allocated);
        curr = curr->next;
      }
    }
  }

  // checkpoint the state of an allocation that may have been
  // split into multiple chunks
  void setSegmentStateToCheckpoint(
      Chunk* chunk,
      SegmentState& segment,
      std::shared_ptr<c10::GatheredContext> context,
      RestoreResult& rr) {
    Chunk* curr_chunk = chunk;
    Chunk* last_chunk = chunk;

    TORCH_INTERNAL_ASSERT(chunk->pool);
    ChunkPool& pool = *chunk->pool;
    const auto segment_len = segment.chunks.size();

    // allocate all chunks in the segment
    for (size_t i = 0; i < segment_len; ++i) {
      auto& chunk_state = segment.chunks.at(i);
      AllocParams params(
          chunk_state.device,
          chunk_state.size,
          chunk_state.stream,
          &pool,
          chunk_state.size,
          stats);
      pool.chunks.erase(curr_chunk);
      params.chunk = curr_chunk;
      params.stat_types = get_stat_types_for_pool(pool);

      // splitting a chunk depends on `max_split_size`, which may have changed
      // between whe checkpoint was taken and now, so we make sure to recreate
      // the behavior from the checkpoint.
      bool split = (i + 1) < segment.chunks.size();

      // curr_chunk will become next pointer if it is split, so reassign with
      // the returned value
      curr_chunk = alloc_found_chunk(
          std::move(params), chunk_state.size, context, split);

      TORCH_CHECK(curr_chunk->ptr == chunk_state.ptr);
      TORCH_CHECK(curr_chunk->size == chunk_state.size);

      last_chunk = curr_chunk;
      curr_chunk = curr_chunk->next;

      TORCH_CHECK((curr_chunk != nullptr) == ((i + 1) < (segment_len)));
    }

    while (last_chunk->prev) {
      last_chunk = last_chunk->prev;
    }

    // free chunks that are not allocated in the checkpoint
    curr_chunk = last_chunk;

    for (size_t i = 0; i < segment_len; ++i, curr_chunk = curr_chunk->next) {
      auto& chunk_state = segment.chunks.at(i);
      TORCH_INTERNAL_ASSERT(curr_chunk != nullptr);

      if (chunk_state.allocated) {
        rr.allocations_created.push_back(curr_chunk);
        continue;
      }

      free(curr_chunk);

      TORCH_CHECK(curr_chunk->ptr == chunk_state.ptr);
      TORCH_CHECK(curr_chunk->allocated == chunk_state.allocated);
      TORCH_CHECK(curr_chunk->size == chunk_state.size);
    }
  }

  RestoreResult setCheckpointPoolState(PrivatePoolState& pps) {
    // To reset the caching allocator state we will
    // - Free all the chunks currently allocated to the pool (see [live tensors
    // between iterations])
    // - Allocate all the chunks in a checkpointed segment, whether they are
    // live or not
    // - Free the chunks in a checkpointed segment which are not live
    // This could be optimized, but it nicely reuses exiting apis, and this
    // is not on the hot path.

    // following `done outside the lock because we don't know what locks the
    // recorder needs to have...`

    std::shared_ptr<c10::GatheredContext> context =
        maybeGatherContext(RecordContext::STATE);

    std::lock_guard<std::recursive_mutex> lock(mutex);

    RestoreResult rr;

    TORCH_CHECK(
        !graph_pools_freeable.count(pps.owner_id),
        "Not expected to checkpoint freeable graph");

    auto pool = graph_pools.find(pps.owner_id);
    TORCH_CHECK(pool != graph_pools.end(), "Could not find private pool id");

    PrivatePool* private_pool = pool->second.get();

    freeChunksAllocatedToPool(private_pool, rr);

    std::unordered_map<void*, Chunk*> ptrs_to_chunks;
    // at this point, all of the chunks should be free, so they will all be in
    // the chunk set
    for (Chunk* chunk : private_pool->small_chunks.chunks) {
      ptrs_to_chunks[chunk->ptr] = chunk;
    }
    for (Chunk* chunk : private_pool->large_chunks.chunks) {
      ptrs_to_chunks[chunk->ptr] = chunk;
    }

    for (auto& segment : pps.segments) {
      auto ptr = segment.chunks.at(0).ptr;
      TORCH_CHECK(ptrs_to_chunks.count(ptr), " could not find ", ptr)
      auto chunk = ptrs_to_chunks[ptr];

      setSegmentStateToCheckpoint(chunk, segment, context, rr);
    }
    return rr;
  }

  /** Dump a complete snapshot of the memory held by the allocator. Potentially
   * VERY expensive. **/
  std::vector<SegmentInfo> snapshot() {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    std::unordered_map<PrivatePool*, MempoolId_t> pool_to_id;
    pool_to_id.reserve(graph_pools.size() + graph_pools_freeable.size());
    for (const auto& pair : graph_pools) {
      pool_to_id[pair.second.get()] = pair.first;
    }
    for (const auto& pair : graph_pools_freeable) {
      pool_to_id[pair.second] = pair.first;
    }

    size_t total_active = 0;
    std::vector<SegmentInfo> result;
    const auto all_chunks = get_all_chunks();

    for (const Chunk* const head_chunk : all_chunks) {
      if (head_chunk->prev != nullptr && head_chunk->prev->mapped) {
        continue;
      }
      result.emplace_back();
      SegmentInfo& segment_info = result.back();
      segment_info.device = head_chunk->device;
      segment_info.address = reinterpret_cast<int64_t>(head_chunk->ptr);
      segment_info.stream = head_chunk->stream;
      segment_info.is_large = (!head_chunk->pool->is_small);
      segment_info.is_expandable = head_chunk->expandable_segment_;
      segment_info.context_when_allocated =
          head_chunk->context_when_segment_allocated;
      auto mempool_id = pool_to_id.find(head_chunk->pool->owner_PrivatePool);
      if (mempool_id != pool_to_id.end()) {
        segment_info.owner_private_pool_id = mempool_id->second;
      }

      const Chunk* chunk = head_chunk;
      while (chunk != nullptr && chunk->mapped) {
        segment_info.chunks.emplace_back();
        ChunkInfo& chunk_info = segment_info.chunks.back();

        chunk_info.size = chunk->size;
        chunk_info.requested_size = chunk->requested_size;
        chunk_info.allocated = chunk->allocated;
        chunk_info.active = chunk->allocated || (chunk->event_count > 0) ||
            !chunk->stream_uses.empty();

        segment_info.total_size += chunk_info.size;
        if (chunk_info.allocated) {
          segment_info.allocated_size += chunk_info.size;
        }
        if (chunk_info.active) {
          segment_info.active_size += chunk_info.size;
          segment_info.requested_size += chunk_info.requested_size;
        }
        chunk_info.context_when_allocated = chunk->context_when_allocated;
        chunk = chunk->next;
      }
      total_active += segment_info.active_size;
    }

    std::sort(
        result.begin(),
        result.end(),
        [](const SegmentInfo& a, const SegmentInfo& b) {
          if (a.device != b.device) {
            return a.device < b.device;
          }
          return a.address < b.address;
        });

    record_trace(TraceEntry::SNAPSHOT, 0, total_active, nullptr, 0, nullptr);
    return result;
  }

  std::vector<TraceEntry> trace() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    std::vector<TraceEntry> result;
    result.reserve(alloc_trace->size());
    result.insert(
        result.end(),
        alloc_trace->begin() + alloc_trace_next,
        alloc_trace->end());
    result.insert(
        result.end(),
        alloc_trace->begin(),
        alloc_trace->begin() + alloc_trace_next);

    return result;
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (C10_UNLIKELY(size <= 4 || divisions <= 1)) {
      return size;
    }
    if (c10::llvm::isPowerOf2_64(size)) {
      return size;
    }

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = c10::llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - c10::llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }

  size_t roundUpSize(size_t size) {
    if (size < minimum_round_size) {
      return minimum_round_size;
    } else if (
        supports_linear_memory && MLUAllocatorConfig::use_linear_memory()) {
      // Linear memory alignment constraints:
      // 1. When requesting a size less than 1GB, alignment must be to a power
      // of 2.
      // 2. When requesting a size greater than 1GB, alignment must be to 1GB.
      const size_t gigabyte = 1 << 30;
      if (size < gigabyte) {
        return c10::llvm::PowerOf2Ceil(size);
      } else {
        return (size + gigabyte - 1) & ~(gigabyte - 1);
      }
    } else {
      auto divisions = MLUAllocatorConfig::roundup_power2_divisions(size);
      if (divisions > 0 && size > (minimum_round_size * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return minimum_round_size *
            ((size + minimum_round_size - 1) / minimum_round_size);
      }
    }
  }

  // Called by MLUGraph::capture_begin
  void beginAllocateToPool(
      MempoolId_t mempool_id,
      std::function<bool(cnrtQueue_t)> filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool. Make a new pool for
      // this capture.
      graph_pools.emplace(mempool_id, std::make_unique<PrivatePool>());
    } else {
      // mempool_id references an existing pool, which the current capture will
      // share. Check this pool is live (at least one other capture already
      // references it).
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      it->second->use_count++;
    }
    for (auto it2 = captures_underway.begin(); it2 != captures_underway.end();
         ++it2) {
      TORCH_CHECK(
          it2->first != mempool_id,
          "beginAllocateToPool: already recording to mempool_id");
    }
    captures_underway.emplace_back(mempool_id, std::move(filter));
  }

  // Called by MLUGraph::capture_end
  void endAllocateToPool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    for (auto it = captures_underway.begin(); it != captures_underway.end();
         ++it) {
      if (it->first == mempool_id) {
        captures_underway.erase(it);
        return;
      }
    }
    TORCH_CHECK(
        false, "endAllocatePool: not currently recording to mempool_id");
  }

  // Called by MLUGraph::reset
  void releasePool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // The instantiated cnrtTaskTopoEntity_t has been destroyed. We can't
    // blindly delete and cnrtFree the mempool its capture used, because
    //  1. other graph(s) might share the same pool
    //  2. the user might still hold references to output tensors allocated
    //  during capture.
    // To handle 1 and 2, we track the number of graphs using this particular
    // mempool. When the count reaches 0, we tell free_cached_chunks it may now
    // cnrtFree chunks from this graph's pool when it discovers they're unused
    // (unsplit).
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    auto uc = --(it->second->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      // Allows free_cached_chunks to begin cnrtFreeing this pool's memory,
      // and makes sure this pool wasn't somehow made freeable already.
      bool inserted =
          graph_pools_freeable.insert({mempool_id, it->second.get()}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }

  void addPeerAccess(int dev_to_access) {
    if (std::find(
            devices_with_peer_access_.begin(),
            devices_with_peer_access_.end(),
            dev_to_access) != devices_with_peer_access_.end()) {
      return;
    }
    devices_with_peer_access_.push_back(dev_to_access);
    for (auto& es : expandable_segments_) {
      es->addPeer(dev_to_access);
    }
  }

  bool hasAllocatedExpandableSegments() const {
    return !expandable_segments_.empty();
  }

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<const Chunk*> get_all_chunks() const {
    std::vector<const Chunk*> chunks;
    chunks.insert(
        chunks.end(), small_chunks.chunks.begin(), small_chunks.chunks.end());
    chunks.insert(
        chunks.end(), large_chunks.chunks.begin(), large_chunks.chunks.end());
    for (const auto& gp : graph_pools) {
      chunks.insert(
          chunks.end(),
          gp.second->small_chunks.chunks.begin(),
          gp.second->small_chunks.chunks.end());
      chunks.insert(
          chunks.end(),
          gp.second->large_chunks.chunks.begin(),
          gp.second->large_chunks.chunks.end());
    }
    chunks.insert(chunks.end(), active_chunks.begin(), active_chunks.end());
    return chunks;
  }

  std::vector<Chunk*> get_private_pool_head_chunks(PrivatePool* pool) const {
    std::vector<Chunk*> chunks;
    for (Chunk* b : active_chunks) {
      if ((b->pool == &pool->small_chunks || b->pool == &pool->large_chunks) &&
          b->prev == nullptr) {
        chunks.push_back(b);
      }
    }

    for (Chunk* b : pool->small_chunks.chunks) {
      if (b->prev == nullptr) {
        chunks.push_back(b);
      }
    }
    for (Chunk* b : pool->large_chunks.chunks) {
      if (b->prev == nullptr) {
        chunks.push_back(b);
      }
    }

    return chunks;
  }

  // returns the smallest possible address in any segment
  // where there is enough free address space to fit size
  // may be composed of free and unmapped segments
  Chunk* find_expandable_chunk(
      int device,
      cnrtQueue_t stream,
      ChunkPool* pool,
      size_t size) {
    // TODO(lipenghui): add expandable segment support.
    return nullptr;
  }

  bool map_chunk(
      Chunk* to_map,
      size_t size,
      const std::shared_ptr<c10::GatheredContext>& ctx) {
    // TODO(lipenghui): add expandable segment support.
    return false;
  }

  Chunk* try_allocate_expandable_chunk(
      int device,
      cnrtQueue_t stream,
      ChunkPool* pool,
      size_t size,
      const std::shared_ptr<c10::GatheredContext>& ctx) {
    // TODO(lipenghui): add expandable segment support.
    return nullptr;
  }

  /** moves a chunk into a pool of cached free chunks */
  void freeChunk(
      Chunk* chunk,
      const std::shared_ptr<c10::GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(
        !chunk->allocated && chunk->event_count == 0 &&
        chunk->stream_uses.empty());

    record_trace(
        TraceEntry::FREE_COMPLETED,
        int64_t(chunk->ptr),
        chunk->requested_size,
        chunk->stream,
        chunk->device,
        context ? context : chunk->context_when_allocated);

    chunk->context_when_allocated = nullptr;
    size_t original_chunk_size = chunk->size;
    size_t requested_size = chunk->requested_size;

    auto& pool = *chunk->pool;
    int64_t net_change_inactive_split_chunks = 0;
    int64_t net_change_inactive_split_size = 0;

    const std::array<Chunk*, 2> merge_candidates = {chunk->prev, chunk->next};
    for (Chunk* merge_candidate : merge_candidates) {
      const int64_t subsumed_size = mergeChunks(chunk, merge_candidate, pool);
      if (subsumed_size > 0) {
        net_change_inactive_split_chunks -= 1;
        net_change_inactive_split_size -= subsumed_size;
      }
    }

    active_chunks.erase(chunk);
    // Makes sure the Chunk* isn't already present in the pool we're freeing it
    // back into.
    bool inserted = pool.insert_into_chunks(chunk).second;
    TORCH_INTERNAL_ASSERT(inserted);

    if (chunk->is_split()) {
      net_change_inactive_split_chunks += 1;
      net_change_inactive_split_size += chunk->size;
    }

    StatTypes stat_types = get_stat_types_for_pool(pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      // inactive_split tries to capture the idea that chunks
      // cannot be freed when requested, but fully free pages
      // of expandable chunks can always be freed.
      // The logic to track this as statistic is pretty involved,
      // so we simply just exclude expandable segments from
      // inactive_split
      if (!chunk->expandable_segment_) {
        update_stat(
            stats.inactive_split[stat_type], net_change_inactive_split_chunks);
        update_stat(
            stats.inactive_split_bytes[stat_type],
            net_change_inactive_split_size);
      }
      update_stat(stats.active[stat_type], -1);
      update_stat(
          stats.active_bytes[stat_type],
          -static_cast<std::int64_t>(original_chunk_size));
      update_stat(
          stats.requested_bytes[stat_type],
          -static_cast<std::int64_t>(requested_size));
    });
  }

  /** combine previously split chunks. returns the size of the subsumed chunk,
   * or 0 on failure. */
  size_t mergeChunks(Chunk* dst, Chunk* src, ChunkPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty() || dst->mapped != src->mapped) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
      dst->context_when_segment_allocated =
          std::move(src->context_when_segment_allocated);
    } else { // [dst src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }

    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    auto erased =
        src->mapped ? pool.chunks.erase(src) : pool.unmapped.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  ChunkPool& get_pool(size_t size, cnrtQueue_t stream) {
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only non-empty if some thread has begun and not yet ended
    // a capture, so it's usually 0, and we can short-circuit
    // cudaStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(captures_underway.size())) {
      for (auto& entry : captures_underway) {
        if (entry.second(stream)) {
          auto it1 = graph_pools.find(entry.first);
          TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
          if (size <= small_allocation_size) {
            return it1->second->small_chunks;
          } else {
            return it1->second->large_chunks;
          }
        }
      }
    }

    if (size <= small_allocation_size) {
      return small_chunks;
    } else {
      return large_chunks;
    }
  }

  StatTypes get_stat_types_for_pool(const ChunkPool& pool) {
    StatTypes stat_types = {false};
    stat_types[static_cast<size_t>(StatType::AGGREGATE)] = true;
    stat_types[static_cast<size_t>(
        pool.is_small ? StatType::SMALL_POOL : StatType::LARGE_POOL)] = true;
    return stat_types;
  }

  bool shouldSplit(Chunk* chunk, size_t size) {
    size_t remaining = chunk->size - size;
    if (chunk->pool->is_small || MLUAllocatorConfig::expandable_segments()) {
      return remaining >= minimum_round_size;
    } else {
      return (size < MLUAllocatorConfig::max_split_size()) &&
          (remaining > small_allocation_size);
    }
  }

  // get allocation size
  size_t getAllocationSize(size_t size) {
    if (size <= small_allocation_size_mlu) { // size <= 16MiB, allocate 32MiB
      return small_buffer_size_mlu;
    } else {
      if (size <
          large_allocation_size_mlu) { // 16MiB < size < 32MiB , allocate 64MiB
        return large_buffer_size_mlu;
      } else if (
          supports_linear_memory && MLUAllocatorConfig::use_linear_memory()) {
        return size;
      } else {
        return maximum_round_size *
            ((size + maximum_round_size - 1) / maximum_round_size);
      }
    }
  }

  bool getFreeChunk(AllocParams& p) {
    ChunkPool& pool = *p.pool;

    if (C10_UNLIKELY(
            set_fraction &&
            MLUAllocatorConfig::garbage_collection_threshold() > 0.0)) {
      // Track chunk reuse interval only when garbage collection is enabled.
      ++pool.get_free_chunks_call_count;
    }
    auto it = pool.chunks.lower_bound(&p.search_key);
    if (it == pool.chunks.end() || (*it)->stream != p.stream())
      return false;

    if ((*it)->expandable_segment_) {
      // TODO(lipenghui): add expandable segment support.
    }

    // Do not return an oversized chunk for a large request
    if ((p.size() < MLUAllocatorConfig::max_split_size()) &&
        ((*it)->size >= MLUAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized chunk size to be rounded up but within a limit
    if ((p.size() >= MLUAllocatorConfig::max_split_size()) &&
        ((*it)->size >= p.size() + large_buffer_size_mlu))
      return false;
    p.chunk = *it;
    pool.chunks.erase(it);
    return true;
  }

  bool trigger_free_memory_callbacks(AllocParams& p) {
    bool freed_memory = false;
    for (const auto& name : FreeMluMemoryCallbacksRegistry()->Keys()) {
      freed_memory |= FreeMluMemoryCallbacksRegistry()->Create(name)->Execute();
    }
    return freed_memory;
  }

  void garbage_collect_cached_chunks(
      const std::shared_ptr<c10::GatheredContext>& context) {
    // Free unused cached chunks to reclaim MLU memory.
    // Unlike free_cached_chunks(), this does not enforce synchronization and
    // therefore should be of less overheads.

    size_t gc_threshold = static_cast<size_t>(
        MLUAllocatorConfig::garbage_collection_threshold() *
        allowed_memory_maximum);
    // No need to trigger GC yet
    if (total_allocated_memory <= gc_threshold) {
      return;
    }
    const auto target_size = total_allocated_memory - gc_threshold;
    size_t gc_reclaimed = 0;

    // Calculate the total age of the free-able chunks. We'll use it later to
    // get "avg age" threshold.
    double total_age = 0.0;
    int freeable_chunk_count = 0;
    for (auto& b : large_chunks.chunks) {
      if (!b->is_split()) {
        total_age += b->gc_count();
        ++freeable_chunk_count;
      }
    }
    // No free-able chunks?
    if (freeable_chunk_count == 0) {
      return;
    }

    // Repeat GC until we reach reclaim > target size.
    bool chunk_freed = true;
    while (gc_reclaimed < target_size && chunk_freed == true &&
           freeable_chunk_count > 0) {
      // Free chunks exceeding this age threshold first.
      double age_threshold = total_age / freeable_chunk_count;
      // Stop iteration if we can no longer free a chunk.
      chunk_freed = false;

      // Free chunks of > avg age. Don't stop upon reaching the target_size,
      // we don't want this GC to be triggered frequently.
      auto it = large_chunks.chunks.begin();
      while (it != large_chunks.chunks.end()) {
        Chunk* chunk = *it;
        ++it;
        if (!chunk->is_split() && chunk->gc_count() >= age_threshold) {
          chunk_freed = true;
          gc_reclaimed += chunk->size;
          total_age -= chunk->gc_count(); // Decrement the age
          freeable_chunk_count--; // One less chunk that can be freed
          releaseChunk(chunk, context);
        }
      }
    }
  }

  bool alloc_chunk(
      AllocParams& p,
      bool isRetry,
      const std::shared_ptr<c10::GatheredContext>& ctx) {
    // Defensively checks for preexisting CNRT error state.
    TORCH_CNRT_CHECK(cnrtGetLastError());

    size_t alloc_size = p.alloc_size;
    void* ptr;

    if (isRetry) {
      stats.num_alloc_retries += 1;
    }

    if (set_fraction &&
        total_allocated_memory + alloc_size > allowed_memory_maximum) {
      p.err = cnrtErrorNoMem;
      return false;
    } else if (MLUAllocatorConfig::expandable_segments()) {
      // TODO(lipenghui): add expandable segments support.
    } else {
      p.err = cnrtMallocMaybeCapturing(&ptr, alloc_size);
      if (p.err != cnrtSuccess) {
        if ((p.err == cnrtErrorNoMem) || (p.err == cnrtErrorCndrvFuncCall)) {
          // when oom happens, the alloc_chunk will return nullptr, handle it
          // outside this function.
          cnrtGetLastError(); // clear MLU error
        } else {
          // If the error's unrelated to memory allocation, we should throw
          // immediately.
          TORCH_CNRT_CHECK(p.err);
        }
        return false;
      }
    }

    if (p.pool->owner_PrivatePool) {
      // The chunk is for a MLU graph's PrivatePool.
      p.pool->owner_PrivatePool->cnrtMalloc_count++;
    }

    total_allocated_memory += alloc_size;
    p.chunk = new Chunk(p.device(), p.stream(), alloc_size, p.pool, (char*)ptr);
    for_each_selected_stat_type(p.stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], 1);
      update_stat(stats.reserved_bytes[stat_type], alloc_size);
    });
    if (alloc_size >= MLUAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, 1);

    // p.chunk came from new cnrtMalloc. It should not be nullptr here.
    TORCH_INTERNAL_ASSERT(p.chunk != nullptr && p.chunk->ptr != nullptr);
    record_trace(
        TraceEntry::SEGMENT_ALLOC,
        int64_t(p.chunk->ptr),
        p.chunk->size,
        p.stream(),
        p.device(),
        ctx);
    p.chunk->context_when_segment_allocated = ctx;
    return true;
  }

  /* Free one or more oversize chunks to the system allocator. But only enough
   * to satisfy the target size
   * */
  bool free_available_cached_chunks(
      const AllocParams& p,
      const std::shared_ptr<c10::GatheredContext>& context) {
    if (MLUAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    ChunkPool& pool = *p.pool;

    // because of std::unique_ptr, chunk cannot be trivially copied
    // Use constructor for search key.
    Chunk key(p.search_key.device, p.search_key.stream, p.search_key.size);
    key.size = (key.size < MLUAllocatorConfig::max_split_size())
        ? MLUAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.chunks.lower_bound(&key);
    if (it == pool.chunks.end() || (*it)->stream != p.stream()) {
      // no single chunk is large enough; free multiple oversize chunks,
      // starting with largest
      if (it == pool.chunks.begin())
        return false;
      size_t total_freed = 0;
      --it; // backup one item。 Now on the largest chunk for the correct
            // stream.
      while ((total_freed < key.size) &&
             ((*it)->size >= MLUAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        total_freed += (*it)->size;
        if (it != pool.chunks.begin()) {
          --it;
          releaseChunk(*cur, context);
        } else {
          releaseChunk(*cur, context);
          break;
        }
      }
      if (total_freed < key.size)
        return false;
    } else {
      releaseChunk(*it, context);
    }
    return true;
  }

  bool free_cached_chunks(
      const std::shared_ptr<c10::GatheredContext>& context) {
    // First ensure that all chunks that can't currently be allocated due to
    // outstanding events are returned to the pool.
    synchronize_and_free_event(context);

    releaseChunks(large_chunks, context);
    releaseChunks(small_chunks, context);

    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      releaseChunks(it->second->small_chunks, context);
      releaseChunks(it->second->large_chunks, context);
      if (it->second->cnrtMalloc_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }

    return true;
  }

  void release_expandable_segment(Chunk* chunk) {
    // TODO(lipenghui): add expandable segments support.
  }

  void releaseChunk(
      Chunk* chunk,
      const std::shared_ptr<c10::GatheredContext>& context) {
    TORCH_INTERNAL_ASSERT(!chunk->expandable_segment_);
    record_trace(
        TraceEntry::SEGMENT_FREE,
        int64_t(chunk->ptr),
        chunk->size,
        chunk->stream,
        chunk->device,
        context ? context : chunk->context_when_segment_allocated);
    // cudaFree is implicit synchronizes the device before freeing the memory.
    // cnrtFree doesn't support this function, so sync device before call
    // cnrtFree.
    TORCH_CNRT_CHECK(cnrtSyncDevice());
    // std::lock_guard<std::mutex> lock(mlu_free_mutex);
    TORCH_CNRT_CHECK(cnrtFree((void*)chunk->ptr));
    total_allocated_memory -= chunk->size;

    auto* pool = chunk->pool;
    if (pool->owner_PrivatePool) {
      // The cnrtFreed chunk belonged to a MLU graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->cnrtMalloc_count > 0);
      pool->owner_PrivatePool->cnrtMalloc_count--;
    }
    StatTypes stat_types = get_stat_types_for_pool(*pool);
    for_each_selected_stat_type(stat_types, [&](size_t stat_type) {
      update_stat(stats.segment[stat_type], -1);
      update_stat(
          stats.reserved_bytes[stat_type],
          -static_cast<std::int64_t>(chunk->size));
    });

    if (chunk->size >= MLUAllocatorConfig::max_split_size())
      update_stat(stats.oversize_segments, -1);

    pool->chunks.erase(chunk);
    delete chunk;
  }

  void unmap_chunk(Chunk* chunk) {
    // TODO(lipenghui): add expandable segments support.
  }

  void releaseChunks(
      ChunkPool& pool,
      const std::shared_ptr<c10::GatheredContext>& context) {
    std::vector<Chunk*> to_unmap;
    // Frees all non-split chunks
    auto it = pool.chunks.begin();
    while (it != pool.chunks.end()) {
      Chunk* chunk = *it;
      ++it;
      if (chunk->expandable_segment_) {
        // unmapping will mutate the free pool
        // so just gather what needs to be freed
        // to avoid invalidating the iterator
        to_unmap.push_back(chunk);
      } else if (!chunk->prev && !chunk->next) {
        releaseChunk(chunk, context);
      }
    }
    for (Chunk* chunk : to_unmap) {
      unmap_chunk(chunk);
      if (!chunk->prev && !chunk->next) {
        release_expandable_segment(chunk);
      }
    }
  }

  void synchronize_and_free_event(
      const std::shared_ptr<c10::GatheredContext>& context) {
    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway.size() == 0);
    insert_events_deferred_until_no_capture();

    for (auto& q : events) {
      for (auto& n : q.second) {
        auto event_sptr = n.first;
        Chunk* chunk = n.second;
        event_sptr->synchronize();
        MLUEventPool_Manager.give_back_event(event_sptr);
        chunk->event_count--;
        if (chunk->event_count == 0) {
          freeChunk(chunk, context);
        }
      }
    }

    events.clear();
  }

  void insert_events(Chunk* chunk) {
    int prev_device = 0;
    TORCH_CNRT_CHECK(cnrtGetDevice(&prev_device));

    stream_set streams(std::move(chunk->stream_uses));
    AT_ASSERT(chunk->stream_uses.empty());
    for (auto& stream : streams) {
      TORCH_CNRT_CHECK(cnrtSetDevice(stream.device_index()));

      c10::DeviceIndex device_id =
          static_cast<c10::DeviceIndex>(stream.device_index());
      auto event_sptr = MLUEventPool_Manager.alloc_event(device_id);
      event_sptr->place(stream);
      chunk->event_count++;
      events[stream].emplace_back(event_sptr, chunk);
    }

    TORCH_CNRT_CHECK(cnrtSetDevice(prev_device));
  }

  void insert_events_deferred_until_no_capture() {
    if (C10_UNLIKELY(!needs_events_deferred_until_no_capture.empty())) {
      for (auto* chunk : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!chunk->stream_uses.empty());
        insert_events(chunk);
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  void process_events(const std::shared_ptr<c10::GatheredContext>& context) {
    insert_events_deferred_until_no_capture();

    // Process outstanding MLU events. MLUEvents that are completed are
    // removed from the stream, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = events.begin(); it != events.end();) {
      while (!it->second.empty()) {
        auto& n = it->second.front();
        auto event_sptr = n.first;
        Chunk* chunk = n.second;
        torch_mlu::mlu::MLUGuard guard(event_sptr->device_index());
        const bool ret = event_sptr->query();
        if (ret == false) {
          // ignore and clear the error if not ready
          cnrtGetLastError();
          break;
        }
        MLUEventPool_Manager.give_back_event(event_sptr);
        chunk->event_count--;
        if (chunk->event_count == 0) {
          freeChunk(chunk, context);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = events.erase(it);
      } else {
        it++;
      }
    }
  }

  // Accumulates sizes of all memory chunks for given device in given pool
  void cache_info_aux(const ChunkPool& pool, size_t* largest) {
    for (const auto& chunk : pool.chunks) {
      size_t chunksize = chunk->size;
      if (chunksize > *largest) {
        *largest = chunksize;
      }
    }
  }

  void record_trace(
      TraceEntry::Action action,
      int64_t addr,
      size_t size,
      cnrtQueue_t stream,
      int device,
      std::shared_ptr<c10::GatheredContext> context) {
    if (!record_history && !trace_trackers_.size())
      return;

    auto te = TraceEntry(
        action,
        device,
        addr,
        size,
        stream,
        record_context_ >= RecordContext::ALLOC ? std::move(context) : nullptr);

    // Callbacks should not include any Pytorch call
    for (const auto& cb : trace_trackers_) {
      cb(te);
    }

    if (record_history) {
      if (alloc_trace->size() < alloc_trace_max_entries_) {
        alloc_trace->emplace_back(te);
      } else {
        (*alloc_trace)[alloc_trace_next++] = te;
        if (alloc_trace_next == alloc_trace_max_entries_) {
          alloc_trace_next = 0;
        }
      }
    }
  }
};

// Returns whether to force all allocations to bypass the caching allocator and
// go straight to cnrtMalloc.  This setting is useful when debugging MLU memory
// errors, since the caching allocator foils cnrt-memcheck.
bool forceUncachedAllocator() {
  static char* force_uncached = getenv("PYTORCH_NO_MLU_MEMORY_CACHING");
  if (force_uncached == nullptr)
    return false;
  auto env_str = std::string(force_uncached);
  return env_str != "0" && env_str != "NO" && env_str != "OFF";
}

static void uncached_delete(void* ptr) {
  if (TORCH_SDT_IS_ENABLED(free)) {
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }
  // TODO(lipenghui): trace mlu memory deallocation
  TORCH_CNRT_CHECK(cnrtFree(ptr));
}

void local_raw_delete(void* ptr);

class NativeCachingAllocator : public MLUAllocator {
 private:
  std::mutex mutex;

  // allocated chunks by device pointer
  ska::flat_hash_map<void*, Chunk*> allocated_chunks;

  void add_allocated_chunk(Chunk* chunk) {
    std::lock_guard<std::mutex> lock(mutex);
    allocated_chunks[chunk->ptr] = chunk;
  }

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocator;

  Chunk* get_allocated_chunk(void* ptr, bool remove = false) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = allocated_chunks.find(ptr);
    if (it == allocated_chunks.end()) {
      return nullptr;
    }
    Chunk* chunk = it->second;
    if (remove) {
      allocated_chunks.erase(it);
    }
    return chunk;
  }

  void init(int device_count) override {
    const auto size = static_cast<int64_t>(device_allocator.size());
    if (size < device_count) {
      device_allocator.resize(device_count);
      for (const auto i : c10::irange(size, device_count)) {
        device_allocator[i] = std::make_unique<DeviceCachingAllocator>();
      }
    }
  }

  bool initialized() override {
    return !device_allocator.empty();
  }

  /** allocates a chunk which is safe to use from the provided stream */
  void malloc(void** devPtr, int device, size_t size, cnrtQueue_t stream) {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    Chunk* chunk = device_allocator[device]->malloc(device, size, stream);
    add_allocated_chunk(chunk);
    *devPtr = (void*)chunk->ptr;
    // TODO(lipenghui): trace mlu memory allocation
  }

  void free(void* ptr) {
    if (!ptr) {
      return;
    }
    Chunk* chunk = get_allocated_chunk(ptr, true /*remove*/);
    if (!chunk) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    // TODO(lipenghui): trace mlu memory deallocation
    device_allocator[chunk->device]->free(chunk);
  }

  void setMemoryFraction(double fraction, int device) override {
    TORCH_INTERNAL_ASSERT(
        0 <= device && static_cast<size_t>(device) < device_allocator.size(),
        "Allocator not initialized for device ",
        device,
        ": did you call init?");
    TORCH_INTERNAL_ASSERT(
        0 <= fraction && fraction <= 1,
        "invalid fraction:",
        fraction,
        ". Please set within (0, 1).");
    int activated_device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&activated_device));
    if (activated_device != device) {
      TORCH_CNRT_CHECK(cnrtSetDevice(device));
    }
    device_allocator[device]->setMemoryFraction(fraction);
  }

  void recordHistory(
      bool enabled,
      CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      RecordContext when) override {
    for (auto& allocator : device_allocator) {
      allocator->recordHistory(
          enabled, context_recorder, alloc_trace_max_entries, when);
    }
  }

  bool isHistoryEnabled() override {
    int device = 0;
    TORCH_CNRT_CHECK(cnrtGetDevice(&device));
    return device_allocator[device]->isHistoryEnabled();
  }

  bool checkPoolLiveAllocations(
      int device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) override {
    return device_allocator[device]->checkPoolLiveAllocations(
        mempool_id, expected_live_allocations);
  }

  void attachAllocatorTraceTracker(AllocatorTraceTracker tracker) override {
    for (auto& allocator : device_allocator) {
      allocator->attachAllocatorTraceTracker(tracker);
    }
  }

  void emptyCache() override {
    for (auto& da : device_allocator) {
      da->emptyCache();
    }
  }

  void* getBaseAllocation(void* ptr, size_t* outSize) override {
    Chunk* chunk = get_allocated_chunk(ptr);
    if (!chunk) {
      TORCH_CHECK(false, "invalid device pointer: ", ptr);
    }
    return device_allocator[chunk->device]->getBaseAllocation(chunk, outSize);
  }

  void recordStream(const c10::DataPtr& ptr, torch_mlu::MLUStream stream)
      override {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // chunks associated with those tensors, it is fine to do nothing here.
    if (!ptr.get()) {
      return;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when MLU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    if (ptr.get_deleter() != &local_raw_delete)
      return;

    Chunk* chunk = get_allocated_chunk(ptr.get());
    TORCH_INTERNAL_ASSERT(chunk != nullptr, "No allocated block can be found");
    device_allocator[chunk->device]->recordStream(chunk, stream);
  }

  SnapshotInfo snapshot() override {
    SnapshotInfo result;
    for (auto& da : device_allocator) {
      result.device_traces.emplace_back(da->trace());
      auto snap = da->snapshot();
      result.segments.insert(result.segments.end(), snap.begin(), snap.end());
    }

    auto& md = result.config_metadata;
    md.garbage_collection_threshold =
        MLUAllocatorConfig::garbage_collection_threshold();
    md.max_split_size = MLUAllocatorConfig::max_split_size();
    md.expandable_segments = MLUAllocatorConfig::expandable_segments();
    md.last_allocator_settings = MLUAllocatorConfig::last_allocator_settings();
    md.roundup_power2_divisions =
        MLUAllocatorConfig::roundup_power2_divisions();

    return result;
  }

  std::shared_ptr<AllocatorState> getCheckpointState(int device, MempoolId_t id)
      override {
    return device_allocator[device]->getCheckpointState(id);
  }

  /**
   * @brief Checkpoint the private pool state identified in `as` to its prior
   * state
   *
   * @param device - device of the pool to manipulate
   * @param as - allocator state
   * @param stale_live_storages - storages of tensors which are currently
   * allocated but which will be not be allocated after the checkpoint is set.
   * For these storages we will remove their deleter function.
   * @return CheckpointDelta - Freed Pointers and DataPtrs that contain deleter
   * functions for all allocated chunks in the new checkpoint state.
   */
  CheckpointDelta setCheckpointPoolState(
      int device,
      std::shared_ptr<AllocatorState> as) override {
    std::shared_ptr<PrivatePoolState> pps =
        std::dynamic_pointer_cast<PrivatePoolState>(as);

    TORCH_CHECK(pps, "Expected PrivatePoolState");

    auto rr = device_allocator[device]->setCheckpointPoolState(*pps);

    CheckpointDelta cpd;
    for (void* ptr : rr.allocations_freed) {
      get_allocated_chunk(ptr, /*remove*/ true);
      cpd.ptrs_freed.push_back(ptr);
    }
    for (Chunk* chunk : rr.allocations_created) {
      add_allocated_chunk(chunk);
      cpd.dataptrs_allocd.emplace_back(
          chunk->ptr,
          chunk->ptr,
          &local_raw_delete,
          at::Device(at::DeviceType::PrivateUse1, device));
    }

    return cpd;
  }

  c10::DataPtr allocate(size_t size) const override {
    if (size == 0) {
      int device;
      TORCH_CNRT_CHECK(cnrtGetDevice(&device));
      return {
          nullptr,
          nullptr,
          c10::detail::deleteNothing,
          c10::Device(c10::DeviceType::PrivateUse1, device)};
    }

    constexpr size_t one_exa_bytes = 1152921504606846976ULL;
    TORCH_CHECK_WITH(
        OutOfMemoryError,
        size < one_exa_bytes,
        "MLU out of memory. Tried to allocate more than 1EB memory.");
    int device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&device));
    void* data = nullptr;
    void (*deleteFunc)(void*) = &local_raw_delete;
    MLUStream stream = torch_mlu::getCurrentMLUStream(device);
    if (forceUncachedAllocator()) {
      deleteFunc = &uncached_delete;

      TORCH_CNRT_CHECK(cnrtMalloc(&data, size));
      // TODO(lipenghui): trace mlu memory allocation
    } else {
      if (size != 0) {
        // Allocator declares allocate const!?
        const_cast<NativeCachingAllocator*>(this)->malloc(
            &data, device, size, stream);
      }
    }

    if (size && TORCH_SDT_IS_ENABLED(malloc)) {
      TORCH_SDT_WITH_SEMAPHORE(malloc, data, device, size, stream.id());
    }

    return {
        data,
        data,
        deleteFunc,
        c10::Device(c10::DeviceType::PrivateUse1, device)};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    if (forceUncachedAllocator()) {
      return &uncached_delete;
    } else {
      return &local_raw_delete;
    }
  }

  void cacheInfo(int dev_id, size_t* largestChunk) override {
    device_allocator[dev_id]->cacheInfo(largestChunk);
  }

  void assertValidDevice(int device) {
    const auto device_num = device_allocator.size();
    TORCH_CHECK(
        0 <= device && device < static_cast<int64_t>(device_num),
        "Invalid device argument ",
        device,
        ": did you call init?");
  }

  MemoryStats getMemoryStats(int device) override {
    assertValidDevice(device);
    return device_allocator[device]->getStats();
  }

  void resetAccumulatedStats(int device) override {
    assertValidDevice(device);
    return device_allocator[device]->resetAccumulatedStats();
  }

  void resetPeakStats(int device) override {
    assertValidDevice(device);
    return device_allocator[device]->resetPeakStats();
  }

  // MLUGraph interactions
  void beginAllocateToPool(
      int device,
      MempoolId_t mempool_id,
      std::function<bool(cnrtQueue_t)> filter) override {
    assertValidDevice(device);
    device_allocator[device]->beginAllocateToPool(
        std::move(mempool_id), std::move(filter));
  }

  void endAllocateToPool(int device, MempoolId_t mempool_id) override {
    assertValidDevice(device);
    device_allocator[device]->endAllocateToPool(mempool_id);
  }

  void releasePool(int device, MempoolId_t mempool_id) override {
    assertValidDevice(device);
    device_allocator[device]->releasePool(std::move(mempool_id));
  }

  void* raw_alloc(size_t nbytes) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, torch_mlu::getCurrentMLUStream(device));
    return r;
  }

  void* raw_alloc_with_stream(size_t nbytes, cnrtQueue_t stream) override {
    if (nbytes == 0) {
      return nullptr;
    }
    int device;
    TORCH_CNRT_CHECK(cnrtGetDevice(&device));
    void* r = nullptr;
    malloc(&r, device, nbytes, stream);
    return r;
  }

  void enablePeerAccess(int dev, int dev_to_access) override {
    torch_mlu::mlu::MLUGuard device_guard(dev);
    unsigned int can_access = 0;
    cnrtRet_t err = cnrtGetPeerAccessibility(&can_access, dev_to_access, 0);
    if (err == cnrtSuccess) {
      // ignore and clear the error if access was already enabled
      (void)cnrtGetLastError();
    } else {
      TORCH_CNRT_CHECK(err);
    }
    if (can_access) {
      device_allocator[dev_to_access]->addPeerAccess(dev);
    }
  }

  cnrtRet_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cnrtQueue_t stream,
      bool p2p_enabled) override {
    if (p2p_enabled || // memcpy ok because memory is mapped in both devices
        srcDevice == dstDevice || // memcpy ok on a single device
        // memcpy ok because both dst and src must have come from cnrtMalloc
        (!device_allocator[dstDevice]->hasAllocatedExpandableSegments() &&
         !device_allocator[srcDevice]->hasAllocatedExpandableSegments())) {
      return cnrtMemcpyAsync(
          dst, const_cast<void*>(src), count, stream, cnrtMemcpyDevToDev);
    }
    // when p2p is not enabled, only cnrtMemcpyPeerAsync correctly handles
    // memory not allocated via cnrtMalloc
    return cnrtMemcpyPeerAsync(
        dst, dstDevice, const_cast<void*>(src), srcDevice, count, stream);
  }

  void raw_delete(void* ptr) override {
    this->free(ptr);
  }

  std::mutex IpcMutex;
  ska::flat_hash_map<std::string, std::weak_ptr<void>> ipcMemHandle_to_devptr;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    std::lock_guard<std::mutex> lock(IpcMutex);
    auto iter = ipcMemHandle_to_devptr.find(handle);
    if (iter != ipcMemHandle_to_devptr.end()) {
      auto devptr = iter->second.lock();
      if (devptr)
        return devptr;
    }
    // This ipcMemHandle hasn't been opened, or already expired, open it to
    // enable IPC access to that mem block.
    void* dev = nullptr;
    auto ipc_handle = reinterpret_cast<const cnrtIpcMemHandle*>(handle.c_str());
    TORCH_CNRT_CHECK(cnrtMapMemHandle(&dev, *ipc_handle, 0));
    // devPtr has to be deleted in same device when created.
    int curr_device = 0;
    TORCH_CNRT_CHECK(cnrtGetDevice(&curr_device));
    auto sp =
        std::shared_ptr<void>(dev, [handle, curr_device, this](void* ptr) {
          torch_mlu::mlu::MLUGuard device_guard(curr_device);
          std::lock_guard<std::mutex> deleter_lock(IpcMutex);
          TORCH_CNRT_CHECK(cnrtUnMapMemHandle(ptr));
          ipcMemHandle_to_devptr.erase(handle);
        });
    std::weak_ptr<void> wp = sp;
    // To eliminate an additional search, we can use insert().
    // It doesn't overwrite when key already exists(ptr expired).
    // But in the deleter for sp we erased the entry,
    // this should be safe to do now.
    ipcMemHandle_to_devptr.insert(iter, {handle, wp});

    return sp;
  }
  std::string name() override {
    return "native";
  }
};

NativeCachingAllocator allocator;

void local_raw_delete(void* ptr) {
  if (TORCH_SDT_IS_ENABLED(free)) {
    TORCH_SDT_WITH_SEMAPHORE(free, ptr);
  }

  allocator.free(ptr);
}
} // namespace Native

// format size(byte) in string
std::string format_size(uint64_t size) {
  std::ostringstream os;
  os.precision(2);
  os << std::fixed;
  if (size <= 1024) {
    os << size << " bytes";
  } else if (size <= 1048576) {
    os << (size / 1024.0);
    os << " KiB";
  } else if (size <= 1073741824ULL) {
    os << size / 1048576.0;
    os << " MiB";
  } else {
    os << size / 1073741824.0;
    os << " GiB";
  }
  return os.str();
}

std::atomic<MLUAllocator*> allocator;

struct BackendStaticInitializer {
  MLUAllocator* parseEnvForBackend() {
    const char* val = getenv("PYTORCH_MLU_ALLOC_CONF");
    if (val != nullptr) {
      const std::string config(val);

      std::regex exp("[\\s,]+");
      std::sregex_token_iterator it(config.begin(), config.end(), exp, -1);
      std::sregex_token_iterator end;
      std::vector<std::string> options(it, end);

      for (auto option : options) {
        std::regex exp2("[:]+");
        std::sregex_token_iterator it2(option.begin(), option.end(), exp2, -1);
        std::sregex_token_iterator end2;
        std::vector<std::string> kv(it2, end2);
        if (kv.size() >= 2) {
          if (kv[0] == "backend") {
            if (kv[1] == "native")
              return &Native::allocator;
          }
        }
      }
    }
    return &Native::allocator;
  }

  BackendStaticInitializer() {
    auto r = parseEnvForBackend();
    allocator.store(r);
  }
};

BackendStaticInitializer backend_static_initializer;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, allocator.load());

MLUAllocator* get() {
  return allocator.load();
}

std::pair<size_t, size_t> MemGetInfo(int device) {
  torch_mlu::mlu::MLUGuard guard(device);
  size_t device_free = 0;
  size_t device_total = 0;
  TORCH_CNRT_CHECK(cnrtMemGetInfo(&device_free, &device_total));
  return {device_free, device_total};
}

// For printing edge memory stats
std::map<std::string, int64_t> mlu_memory_stats(int device) {
  const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
      statTypeNames = {"all", "small_pool", "large_pool"};
  const std::array<const char*, 4> statNames = {
      "current", "peak", "allocated", "freed"};

  const auto statToDict = [](const Stat& stat) {
    std::vector<int64_t> dict(4, 0);
    dict[0] = stat.current;
    dict[1] = stat.peak;
    dict[2] = stat.allocated;
    dict[3] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    std::vector<std::vector<int64_t>> dict;
    for (size_t i = 0; i < statTypeNames.size(); ++i) {
      dict.push_back(statToDict(statArray[i]));
    }
    return dict;
  };

  const MemoryStats stats =
      torch_mlu::MLUCachingAllocator::getMemoryStats(device);
  std::unordered_map<std::string, std::vector<std::vector<int64_t>>> result;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["oversize_allocations"].push_back(
      statToDict(stats.oversize_allocations));
  result["oversize_segments"].push_back(statToDict(stats.oversize_segments));

  std::map<std::string, int64_t> res;
  for (const auto& r : result) {
    for (int i = 0; i < statTypeNames.size(); ++i) {
      if (r.first != "oversize_allocations" && r.first != "oversize_segments") {
        const auto& d = r.second[i];
        std::string out(r.first + "." + statTypeNames[i] + ".");
        for (int j = 0; j < statNames.size(); ++j) {
          res[out + statNames[j]] = d[j];
        }
      } else {
        const auto& d = r.second[0];
        std::string out(r.first + ".");
        for (int j = 0; j < statNames.size(); ++j) {
          res[out + statNames[j]] = d[j];
        }
        continue;
      }
    }
  }
  res["num_alloc_retries"] = stats.num_alloc_retries;
  res["num_ooms"] = stats.num_ooms;

  return res;
}

// return the current memory allocated on MLU
uint64_t currentMemoryAllocated(int dev) {
  TORCH_CHECK(
      dev == -1 || dev >= 0,
      "Device index must be -1 or non-negative, got ",
      dev);
  dev = dev == -1 ? current_device() : dev;
  return mlu_memory_stats(dev)["allocated_bytes.all.current"];
}

// return the current memory cached on MLU
uint64_t currentMemoryCached(int dev) {
  TORCH_CHECK(
      dev == -1 || dev >= 0,
      "Device index must be -1 or non-negative, got ",
      dev);

  dev = dev == -1 ? current_device() : dev;
  return mlu_memory_stats(dev)["reserved_bytes.all.current"];
}

// return the max memory allocated on MLU
uint64_t maxMemoryAllocated(int dev) {
  TORCH_CHECK(
      dev == -1 || dev >= 0,
      "Device index must be -1 or non-negative, got ",
      dev);

  dev = dev == -1 ? current_device() : dev;
  return mlu_memory_stats(dev)["allocated_bytes.all.peak"];
}

// return the max memory cached on MLU
uint64_t maxMemoryCached(int dev) {
  TORCH_CHECK(
      dev == -1 || dev >= 0,
      "Device index must be -1 or non-negative, got ",
      dev);

  dev = dev == -1 ? current_device() : dev;
  return mlu_memory_stats(dev)["reserved_bytes.all.peak"];
}

} // namespace torch_mlu::MLUCachingAllocator
