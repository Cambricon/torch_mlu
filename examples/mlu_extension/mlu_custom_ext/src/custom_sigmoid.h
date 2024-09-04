#pragma once
#include <torch/extension.h>
torch::Tensor active_sigmoid_mlu(torch::Tensor x);
