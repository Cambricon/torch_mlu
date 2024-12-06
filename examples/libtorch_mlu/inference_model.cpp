#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cmath> // for std::abs
#include <torch/library.h>
#include <torch/script.h>

#include "framework/core/device.h"
#include "framework/core/caching_allocator.h"

void assert_close(
    const at::Tensor& a,
    const at::Tensor& b,
    float rtol = 1e-3,
    float atol = 1e-3) {
  if (!a.sizes().equals(b.sizes())) {
    std::cerr << "Error: Tensor shapes are not the same!" << std::endl;
    exit(1);
  }

  auto diff = torch::abs(a - b);
  auto max_diff = diff.max().item<float>();

  if (max_diff > atol + rtol * torch::abs(b).max().item<float>()) {
    std::cerr << "Error: Tensors are not close!" << std::endl;
    std::cerr << "Max diff: " << max_diff << std::endl;
    exit(1);
  }
}

void run_model_in_thread(
    const std::string& pt_model_path,
    int thread_id,
    int loop,
    const at::Tensor& inputs,
    const at::Tensor& cpu_output) {
  // init device
  auto device = at::Device("mlu:0");

  // get mlu out
  torch::jit::script::Module script_model = torch::jit::load(pt_model_path);
  script_model.to(device);
  at::Tensor model_out_mlu;
  for (int i = 0; i < loop; ++i) {
    model_out_mlu = script_model.forward({inputs.to(device)}).toTensor();
  }
  auto model_out_cpu_from_mlu = model_out_mlu.to(at::kCPU);

  // compare mlu an cpu out
  try {
    assert_close(model_out_cpu_from_mlu, cpu_output);
  } catch (const std::exception& e) {
    std::cerr << "Thread " << thread_id << ": MLU and CPU results differ! "
              << e.what() << std::endl;
  }

  std::cout << "Thread " << thread_id << ": Completed inference" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: program <model_path> <num_threads> <loop_count>\n";
    return 0;
  }

  std::string pt_model_path = argv[1];
  int num_threads = std::stoi(argv[2]);
  int loop_count = std::stoi(argv[3]);

  // init device
  auto device = at::Device("mlu:0");

  // create input on device
  auto inputs_mlu = torch::rand({4, 3, 4, 4}).to(device);

  // move input to cpu
  auto inputs_cpu = inputs_mlu.to(at::kCPU);

  // get cpu out
  torch::jit::script::Module script_model_cpu = torch::jit::load(pt_model_path);
  script_model_cpu.to(at::kCPU);
  auto model_out_cpu = script_model_cpu.forward({inputs_cpu}).toTensor();

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(
        run_model_in_thread,
        pt_model_path,
        i,
        loop_count,
        inputs_cpu,
        model_out_cpu);
  }

  for (auto& t : threads) {
    t.join();
  }

  std::cout << "All threads completed." << std::endl;
  return 0;
}
