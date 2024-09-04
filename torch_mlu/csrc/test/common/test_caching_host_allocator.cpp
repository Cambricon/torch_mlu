#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>

#include "ATen/ATen.h"
#include "framework/core/memory_allocator.h"

namespace torch_mlu {

namespace {
constexpr int64_t N = 100;
}

TEST(HostMemoryAllocator, pinned_alias_slice) {
  // Check a standard pinned tensor can be correctly recorded.
  auto pinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  ASSERT_TRUE(pinned_tensor.is_pinned());
  ASSERT_TRUE(torch_mlu::CachingHostAllocator_recordEvent(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      torch_mlu::getCurrentMLUStream()));

  // Check an tensor constructed with from_blob can be correctly recorded (via
  // the shared data_ptr)
  auto alias_tensor = at::from_blob(
      pinned_tensor.data_ptr(), pinned_tensor.sizes(), pinned_tensor.options());
  ASSERT_TRUE(alias_tensor.is_pinned());

  ASSERT_FALSE(
      alias_tensor.storage().data_ptr().get_context() ==
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_EQ(alias_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(torch_mlu::CachingHostAllocator_recordEvent(
      alias_tensor.data_ptr(),
      alias_tensor.storage().data_ptr().get_context(),
      torch_mlu::getCurrentMLUStream()));

  // Check an tensor constructed with slicing can be correctly recorded (via
  // the shared context)
  auto slice_tensor =
      pinned_tensor.index({at::indexing::Slice(1, at::indexing::None, 2)});
  ASSERT_EQ(
      slice_tensor.storage().data_ptr().get_context(),
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_NE(slice_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(torch_mlu::CachingHostAllocator_recordEvent(
      slice_tensor.data_ptr(),
      slice_tensor.storage().data_ptr().get_context(),
      torch_mlu::getCurrentMLUStream()));

  // Check a tensor that has neither a matching context nor data_ptr cannot be
  // recorded.
  auto alias_slice_tensor = at::from_blob(
      slice_tensor.data_ptr(), slice_tensor.sizes(), slice_tensor.options());
  ASSERT_TRUE(alias_slice_tensor.is_pinned());
  ASSERT_FALSE(torch_mlu::CachingHostAllocator_recordEvent(
      alias_slice_tensor.data_ptr(),
      alias_slice_tensor.storage().data_ptr().get_context(),
      torch_mlu::getCurrentMLUStream()));
  ASSERT_NE(
      alias_slice_tensor.storage().data_ptr().get(),
      slice_tensor.storage().data_ptr().get());
}

TEST(HostMemoryAllocator, check_raw_allocation) {
  auto data_ptr = torch_mlu::getCachingHostAllocator()->allocate(N);
  class UserDataDeleter {
   public:
    explicit UserDataDeleter(std::unique_ptr<void, c10::DeleterFnPtr> ptr)
        : ptr_(std::move(ptr)) {}

   private:
    std::unique_ptr<void, c10::DeleterFnPtr> ptr_;
  };
  auto* user_data_deleter = new UserDataDeleter(data_ptr.move_context());

  struct IOBuf {
    explicit IOBuf(void* buf, void* ctx, std::function<void(void*)> deleter)
        : buf_(buf), ctx_(ctx), deleter_(std::move(deleter)) {}
    void* buf_;
    void* ctx_;
    std::function<void(void*)> deleter_;
    ~IOBuf() {
      deleter_(ctx_);
    }
  };
  auto iobuf =
      std::make_unique<IOBuf>(data_ptr.get(), user_data_deleter, [](void* ctx) {
        delete static_cast<UserDataDeleter*>(ctx);
      });
  auto pinned_tensor =
      at::for_blob(iobuf->buf_, {N})
          .context(
              iobuf.release(),
              [](void* ctx) { delete static_cast<IOBuf*>(ctx); })
          .make_tensor();

  ASSERT_TRUE(pinned_tensor.is_pinned());
  ASSERT_TRUE(torch_mlu::CachingHostAllocator_recordEvent(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      torch_mlu::getCurrentMLUStream()));
}

TEST(HostMemoryAllocator, check_unknown_tensor) {
  auto unpinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(false));

  ASSERT_FALSE(torch_mlu::CachingHostAllocator_recordEvent(
      unpinned_tensor.data_ptr(),
      unpinned_tensor.storage().data_ptr().get_context(),
      torch_mlu::getCurrentMLUStream()));
}

TEST(HostMemoryAllocator, check_pin_memory_free_after_record) {
  {
    at::Tensor pin_memory = at::ones(
        {N}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  }
  auto copy_bias_from_pin = [](const at::Tensor& bias) {
    at::Tensor pin_memory = at::ones(
        {N}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
    bias.copy_(pin_memory, true);
  };
  at::Tensor result;
  at::Tensor left = at::ones({N, N}).to("privateuseone");
  at::Tensor right = at::ones({N, N}).to("privateuseone");
  at::Tensor bias = at::zeros({N}).to("privateuseone");
// Push kernel to current queue.
#pragma unroll
  for (int i = 0; i < 5; i++) {
    result = at::matmul(left, right);
  }
  // Pin memory freed after function return.
  copy_bias_from_pin(bias);
  torch_mlu::CachingHostAllocator_emptyCache();
  result.add_(bias);
  at::Tensor result_cpu = result.cpu();
  ASSERT_TRUE(*(result_cpu.data_ptr<float>() + N) == (N + 1));
  torch_mlu::CachingHostAllocator_emptyCache();
}

TEST(HostMemoryAllocator, check_empty_cache) {
  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
    ASSERT_TRUE(torch_mlu::CachingHostAllocator_recordEvent(
        ptr, ctx, torch_mlu::getCurrentMLUStream()));
  }

  // Even this test case without async operations, still get not ready
  // from notify status query sometimes. This will cause this test case failed.
  // So add a time sleep for notify query status ready.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  torch_mlu::CachingHostAllocator_emptyCache();
  ASSERT_FALSE(torch_mlu::CachingHostAllocator_recordEvent(
      ptr, ctx, torch_mlu::getCurrentMLUStream()));
}

TEST(HostMemoryAllocator, check_reuse) {
  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
  }
  // Ensure we reuse the allocation.
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ASSERT_EQ(ptr, pinned_tensor.data_ptr());
    ASSERT_EQ(ctx, pinned_tensor.storage().data_ptr().get_context());
  }
}

} // namespace torch_mlu
