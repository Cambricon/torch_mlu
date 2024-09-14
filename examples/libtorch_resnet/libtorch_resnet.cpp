#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>

#include "framework/core/device.h"
#include "framework/core/caching_allocator.h"

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    TORCH_CHECK(false, "Please input the model name!");
  }

  std::cout << "module: " << argv[1] << std::endl;

  // init mlu
  auto device = at::Device("mlu:0");

  // load model
  torch::jit::script::Module module = torch::jit::load(argv[1]);

  module.to(device);

  // run model
  torch::jit::setGraphExecutorOptimize(false);
  std::vector<torch::jit::IValue> input_tensor;
  input_tensor.push_back(torch::randn({1, 3, 244, 244}).to(device));
  at::Tensor output = module.forward(input_tensor).toTensor();

  std::cout << output.slice(1, 0, 5) << std::endl;
  std::cout << "resnet_model run success!" << std::endl;

  return 0;
}
