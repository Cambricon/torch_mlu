#include <iostream>
#include <torch/library.h>
#include <torch/script.h>

#include "framework/core/MLUEvent.h"
#include "framework/core/MLUStream.h"
#include "framework/core/stream_guard.h"

int main() {
  auto device = at::Device("mlu:0");
  auto t = torch::tensor({1, 2, 3, 4}).pin_memory();
  auto result = at::randn(t.sizes()).to(device);
  auto stream = torch_mlu::getStreamFromPool();
  void* tensor_ptr = nullptr;
  auto input_x =
      at::randn({1024, 1024}, at::kFloat, c10::nullopt, device, c10::nullopt);
  auto input_y =
      at::randn({1024, 1024}, at::kFloat, c10::nullopt, device, c10::nullopt);

  auto event = torch_mlu::MLUEvent();
  {
    at::Tensor tmp;
    {
      torch_mlu::mlu::MLUStreamGuard guard(stream);
      tmp = t.to(device, at::kFloat, true);
      tensor_ptr = tmp.data_ptr();
      event.place(stream);
    }
    // Event to make sure jobs order on two streams.
    event.wait(torch_mlu::getCurrentMLUStream());
    // Use record_stream to make sure device memory belongs to tmp won't be
    // freed when tmp deconstruct. An event E1 will be recorded to current
    // stream when tmp goes out of scope. E1 will be checked when
    // caching allocator tries to malloc.
    tmp.record_stream(torch_mlu::getCurrentMLUStream());
    for (int i = 0; i < 1000; i++) {
      at::matmul(input_x, input_y);
    }
    result.copy_(tmp);
  }
  {
    // Before event E1 is ready, the chunk of memory of tmp won't be reused.
    torch_mlu::mlu::MLUStreamGuard guard(stream);
    auto tmp2 = at::randn(t.sizes()).to(device);
    tmp2.zero_();
    TORCH_CHECK_NE(tmp2.data_ptr(), tensor_ptr);
    std::cout << "Allocation not reused.\n";
  }

  std::cout << "The value of reuslt:\n" << result << "\n";

  // Event E1 is ready, the chunk of memory of tmp will be reused.
  torch_mlu::getCurrentMLUStream().synchronize();
  {
    torch_mlu::mlu::MLUStreamGuard guard(stream);
    auto tmp3 = torch::tensor(t.sizes()).to(device);
    TORCH_CHECK_EQ(tmp3.data_ptr(), tensor_ptr);
    std::cout << "Allocation reused.\n";
  }
  std::cout << "The demo ran successfully.\n";
}
